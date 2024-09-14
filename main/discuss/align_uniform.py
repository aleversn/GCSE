import torch
import torch.nn.functional as F
from transformers import AutoModel

def align_loss(x, y, alpha=2):    
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def get_pair_emb(model, input_ids, attention_mask,token_type_ids):
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    pooler_output = outputs.pooler_output
    pooler_output = pooler_output.view((batch_size, 2, pooler_output.size(-1)))
    z1, z2 = pooler_output[:,0], pooler_output[:,1]
    return z1,z2

def get_align(model, dataloader):
    align_all = []
    unif_all = []
    with torch.no_grad():        
        for data in dataloader:
            input_ids = torch.cat((data['input_ids'][0],data['input_ids'][1])).cuda()
            attention_mask = torch.cat((data['attention_mask'][0],data['attention_mask'][1])).cuda()
            token_type_ids = torch.cat((data['token_type_ids'][0],data['token_type_ids'][1])).cuda()

            z1,z2 = get_pair_emb(model, input_ids, attention_mask, token_type_ids)        
            z1 = F.normalize(z1,p=2,dim=1)
            z2 = F.normalize(z2,p=2,dim=1)

            align_all.append(align_loss(z1, z2, alpha=2))
            
    return align_all

def get_unif(model, dataloader):
    unif_all = []
    with torch.no_grad():        
        for data in dataloader:
            input_ids = torch.cat((data['input_ids'][0],data['input_ids'][1])).cuda()
            attention_mask = torch.cat((data['attention_mask'][0],data['attention_mask'][1])).cuda()
            token_type_ids = torch.cat((data['token_type_ids'][0],data['token_type_ids'][1])).cuda()

            z1,z2 = get_pair_emb(model, input_ids, attention_mask, token_type_ids)        
            z1 = F.normalize(z1,p=2,dim=1)
            z2 = F.normalize(z2,p=2,dim=1)
            z = torch.cat((z1,z2))
            unif_all.append(uniform_loss(z, t=2))

    return unif_all



model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
model = model.cuda()
model_name = "unsup-simcse-bert-base-uncased"

align_all = get_align(model, pos_loader)

align = sum(align_all)/len(align_all)
