import torch
import torch.nn as nn


class UCLIP(nn.Module):
    def __init__(self,
                args,
                clip,
                 ):
        super().__init__()
        self.args = args
        self.clip = clip
        self.tau = args.tau
        self.K = args.K_prototype
        self.embed_dim = self.clip.visual.output_dim

        # vision prototype
        self.v_prototype = nn.parameter.Parameter(torch.zeros(self.K, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.v_prototype) 

        # text prototype
        self.t_prototype = nn.parameter.Parameter(torch.zeros(self.K, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.t_prototype)
        

    def uncertainty_compute(self, sims):
        K = sims.size(1)
        E = torch.exp(sims / self.tau)
        #E = sims
        #E[E < 0] = 0 
        alpha = E + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        evi = K / S

        return sims, 1 - evi
    
    def image_uncertainty_modeling(self, image_feats):
        logit_scale = self.clip.logit_scale.exp()
        t_prototype = self.t_prototype / self.t_prototype.norm(dim=-1,keepdim=True)
        vhub_logits = logit_scale * torch.matmul(image_feats, t_prototype.t())
        v_alpha, vu = self.uncertainty_compute(vhub_logits)

        tt_logits = logit_scale * torch.matmul(t_prototype, t_prototype.t())

        return v_alpha, vu, tt_logits
    
    def text_uncertainty_modeling(self, text_feats):
        logit_scale = self.clip.logit_scale.exp()
        v_prototype = self.v_prototype / self.v_prototype.norm(dim=-1,keepdim=True)
        thub_logits = logit_scale * torch.matmul(text_feats, v_prototype.t())
        t_alpha, tu = self.uncertainty_compute(thub_logits)

        vv_logits = logit_scale * torch.matmul(v_prototype, v_prototype.t())

        return t_alpha, tu, vv_logits

    def forward(self, image, text):
        logit_scale = self.clip.logit_scale.exp()
        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        v_alpha, vu, tt_logits = self.image_uncertainty_modeling(image_features)
        t_alpha, tu, vv_logits = self.text_uncertainty_modeling(text_features)

        # cosine similarity as logits
        i2t_logits = logit_scale * image_features @ text_features.t()
        t2i_logits = i2t_logits.t()

        ret = {}
        ret['v_alpha'] = v_alpha
        ret['t_alpha'] = t_alpha
        # ret['v_prototype'] = v_prototype
        # ret['t_prototype'] = t_prototype
        ret['vv_logits'] = vv_logits
        ret['tt_logits'] = tt_logits
        ret['vu'] = vu
        ret['tu'] = tu

        # shape = [global_batch_size, global_batch_size]
        return i2t_logits, t2i_logits, ret
        