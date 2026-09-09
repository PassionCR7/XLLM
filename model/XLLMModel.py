from transformers import PretrainedConfig


class XLLMConfig(PretrainedConfig):
    model_type = "xllm"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,        # 是否开启MoE，False使用普通FeedForward
        num_experts_per_tok: int = 2, # 每个token激活多少个路由专家(top‑k)
        n_routed_experts: int = 4,    # 路由专家总数量（会被门控选择）
        n_shared_experts: int = 1,    # 共享专家数量
        scoring_func: str = "softmax",# 门控打分函数
        aux_loss_alpha: float = 0.01, # MoE负载均衡辅助损失权重，主loss + alpha * aux_loss
        seq_aux: bool = True,         # 辅助损失模式：True=序列级；False=batch全局级
        norm_topk_prob: bool = True,  # 是否对选中top‑k专家的gate权重做归一化，保证单token权重和=1
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
import math
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

#继承nn.Module 类
class RMSNorm(nn.Module):
#__init__初始化
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim #维度
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim)) #公式的可学习参数γ，构建一个初始化全1、维度dim的参数
#_norm计算
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps) 
#实现forward
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)



def precompute_freqs_cis(
        dim:int,
        end:int=int(32*1024),
        rope_base=1e6,
        rope_scaling:Optional[dict]=None,
):
    #初始化RoPE频率
    freqs,attn_factor=(
        1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 从配置字典中提取 YaRN 的超参数
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max,factor,beta_fast,beta_slow=(
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
            )
        #推理长度大于训练长度就使用缩放
        if end/orig_max>1.0:
           #波长b到i的映射
           inv_dim=lambda b:(dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))
           #划分高低维
           # low为不需要缩放的高频部分
           # high为需要缩放的低频部分
           low,high=(max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2-1))
           #计算缩放因子
           #low不需要缩放，ramp为0，需要缩放的high，ramp为1，两者之间先行过度
           #平滑过渡系数 ramp是一个从 0 线性增长到 1 的斜坡函数
           ramp=torch.clamp(
               (torch.arange(dim//2,device=freqs.device).float()-low)
               / max(high-low, 0.001),
               0,
               1,
           )
           #ramp=0,高频系数是0，保持不变；ramp=1，低频系数是1/factor，对频率进行线性插值缩放。
           freqs=freqs*(1-ramp+ramp/factor)
    #根据end，生成位置索引t（无论是否启用YaRN缩放都要执行）
    t=torch.arange(end,device=freqs.device).float()
    #计算外积，将t和频率部分相乘，得到每个位置的旋转角度
    freqs=torch.outer(t,freqs).float() #每个位置 t、每个维度对 i 的旋转角度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
#编写RoPE
def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    #[a,b]->[-b,a]
    def rotate_half(x):
        #x.shape[-1]取最后一个维度的重点
        #(x[...,x.shape[-1]//2:]取出x的后半部r分
        return torch.cat(
            (-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),dim=-1
        )
    # x_rotated=x*cos+rotate_half(x)*sin
    #对q和k进行旋转位置编码的应用
    q_embed=(q*cos.unsqueeze(unsqueeze_dim)+rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_embed=(k*cos.unsqueeze(unsqueeze_dim)+rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    return q_embed,k_embed



#复制扩充 KV 头，实现 GQA 中多头 Q 共享少量 KV
#输入 shape：[bs, slen, num_kv_heads, head_dim]
#输出 shape与Q头数对齐：[bs, slen, num_kv_heads * n_rep, head_dim]
#head_dim = hidden_size /num_attention_heads
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    bs,slen,num_key_value_heads,head_dim=x.shape
    if n_rep==1: #标准MHA
        return x

    return (
        x[:,:,:,None,:] #在第 3 维插入一个新维度，用于复制扩展
        .expand(bs,slen,num_key_value_heads,n_rep,head_dim)
        .reshape(bs,slen,num_key_value_heads*n_rep,head_dim)
    )

class Attention(nn.Module):
    def __init__(self,args:XLLMConfig):
        super().__init__()
        self.num_key_value_heads=(
            args.num_attention_heads 
            if args.num_key_value_heads is None 
            else args.num_key_value_heads
            )
        #Query 头总数必须能被 KV 头整除,不满足直接抛异常
        assert args.num_attention_heads % self.num_key_value_heads == 0
        "num_attention_heads must be divisible by n_key_value_heads"

        self.n_local_heads=args.num_attention_heads #本地 Query 总头数
        self.n_local_kv_heads=self.num_key_value_heads
        self.n_rep=self.n_local_heads//self.n_local_kv_heads #n_rep：每个 KV 头要复制几份
        self.head_dim=args.hidden_size//args.num_attention_heads #单头维度

        #Llama 系列 transformer 线性层不使用 bias 偏置
        #把所有 query 头的向量全部平铺拼接在一起
        self.q_proj=nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        #多头注意力拼接之后，映射回模型 hidden_size
        self.o_proj=nn.Linear(args.num_attention_heads*self.head_dim,args.hidden_size,bias=False)

        self.attn_dropout=nn.Dropout(args.dropout) #注意力分数矩阵 dropout
        self.resid_dropout=nn.Dropout(args.dropout) #残差连接输出后的 dropout
        self.dropout=args.dropout

        # 判断是否启用 FlashAttention：
        # ）PyTorch 版本是否内置 F.scaled_dot_product_attention(>=2.0)
        # 2）配置打开 flash_attention 开关
        # flash 为 True 走 FlashAttention 加速路径，False 手写原生注意力
        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') and args.flash_attention

    #forward流程：线性投影 →分头 →RoPE 位置编码 →拼接 KV‑Cache →repeat_kv 扩充 KV 头 
    # →FlashAttention / 手写注意力计算 →多头拼接输出，返回 KV 缓存
    def forward(
        self,
        x: torch.Tensor, #shape[bsz,seq_len,hidden_size]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], #RoPE 旋转位置编码返回(cos,sin)
        #KV‑Cache，推理复用历史 K、V；训练时 None
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        #是否返回 KV 缓存，生成推理打开；训练关闭 False
        use_cache=False,
        #padding 掩码，padding 位置屏蔽
        attention_mask: Optional[torch.Tensor] = None,
    ):
        #投影，计算q，k，v
        bsz,seq_len,_=x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x) 
        #线性投影得到原始QKV，k_proj/v_proj的输出维度更小
        #用view把平铺的长向量切分开，每head_dim个元素归为 1 个注意力头，
        # K、V 只拆出少量头，后面就需要调用repeat_kv()复制 KV 头，对齐 Q 头数量
        xq=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim) #xq →[bsz,seq_len,num_q_heads,head_dim]
        xk=xk.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim) #xk/xv →[bsz,seq_len,num_kv_heads,head_dim]
        xv=xv.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        #对q和k使用RoPE
        cos,sin=position_embeddings
        xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin) #RoPE，只作用在q，k上，v不变
        #对k，v使用repeat（注意kv cache）
        if past_key_value is not None: #past_key_value保存上一步所有历史 token 的 K、V
            xk=torch.cat([past_key_value[0],xk],dim=1) #在序列维度dim=1拼接历史 KV 和当前 step 新生成的 KV
            xv=torch.cat([past_key_value[1],xv],dim=1) #不用重新计算全部历史 token，大幅加速生成
        past_kv=(xk,xv) if use_cache else None #use_cache=True才把拼接完的(xk,xv)保存返回给下一轮

        xq,xk,xv=(
            xq.transpose(1,2), #[bsz,seq_len,heads,hd] → [bsz,heads,seq_len,hd]
            repeat_kv(xk,self.n_rep).transpose(1,2), #把 GQA 的少量 KV 头复制扩充，和 Q 头数量对齐
            repeat_kv(xv,self.n_rep).transpose(1,2), #xq/xk/xv统一shape[bsz, num_q_heads, seq_len, head_dim]
        )
        #进行attention注意力分数计算
        if (
            self.flash
            and (seq_len>1)
            and (past_key_value is None)
            ):
            if attention_mask is None or torch.all(attention_mask==1):
                #无 padding：因果掩码交给 is_causal=True，走最快的 FlashAttention 路径
                #注意 attn_mask 与 is_causal 不能同时传入，否则报错
                output=F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    dropout_p=self.dropout if self.training else 0.0, #训练模式启用 dropout；推理 dropout=0
                    is_causal=True, #生成下三角因果掩码
                    )
            else:
                #带 padding 的批次：手工合并"因果掩码+padding掩码"为 4D bool 掩码后传入，
                #同样走高效注意力路径，避免手写分支实例化 [bsz,heads,seq,seq] 分数矩阵导致爆显存
                causal_mask=torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device).tril()
                pad_mask=attention_mask[:,None,None,:].bool() #[bsz,1,1,seq]
                attn_mask=causal_mask[None,None,:,:]&pad_mask #[bsz,1,seq,seq]
                output=F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    )
        else:
            scores=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim) #注意力分数矩阵、缩放
            #只在最新输入的seq_len长度部分施加causal mask，兼容kv cache
            #torch.triu(...,diagonal=1)生成上三角矩阵，对角线以上填充‑inf
            scores[:, :, :, -seq_len:] +=torch.triu(
                torch.full((seq_len,seq_len),float("-inf"),device=scores.device),
                diagonal=1,
                )
            # scores=scores+torch.triu(
            #     torch.full((seq_len,seq_len),float('-inf'),device=scores.device),
            #     diagonal=1,
            # ).unsqueeze(0).unsqueeze(0)
        #最后拼接头，输出投影，返回
            if attention_mask is not None:
                #attention_mask: padding 位置是 0，有效 token 是 1
                #(1.0‑mask)*‑1e9，padding 位置加上负无穷，softmax 之后概率趋近 0
                extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask=(1.0-extended_attention_mask)*-1e9
                scores=scores+extended_attention_mask

            scores=F.softmax(scores.float(),dim=-1).type_as(xq) #dim=-1在 key 序列维度做 softmax
            scores=self.attn_dropout(scores)
            output=scores@xv #注意力权重乘 V，得到多头输出
        #[bsz,n_local_heads,seq_len,head_dim]
        output=output.transpose(1,2).reshape(bsz,seq_len,-1)  #把所有头拼接在一起
        output=self.resid_dropout(self.o_proj(output)) #o_proj输出投影层，把拼接多头输出映射回模型 hidden_size
        return output,past_kv #返回注意力输出、KV 缓存


class FeedForward(nn.Module):
    #初始化
    #升维
    #降维
    #门控
    #dropout
    #激活函数
    def __init__(self,args:XLLMConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size=int(args.hidden_size*8/3)
            args.intermediate_size=64*((intermediate_size+64-1)//64)

        self.up_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.down_proj=nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
        self.gate_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.dropout=nn.Dropout(args.dropout)
        self.act_fn=ACT2FN[args.hidden_act] #SiLU激活函数
    def forward(self,x):
        gated=self.act_fn(self.gate_proj(x))*self.up_proj(x)
        return self.dropout(self.down_proj(gated))


#输入 token 隐状态
#输出：
# 1. topk_idx：每个 token 选中的专家索引
# 2. topk_weight：每个 token 对选中专家对应的 gate 权重
# 3. aux_loss：负载均衡辅助损失
class MoEGate(nn.Module):
    def __init__(self, config: XLLMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        # 打分函数选择
        self.scoring_func = config.scoring_func
        # alpha控制辅助损失的权重：loss + aux_loss*alpha
        self.alpha = config.aux_loss_alpha
        # seq_aux决定是使用句子还是序列的aux_loss
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        # 门控权重矩阵 [n_routed_experts, hidden_size]
        # 等价线性层：hidden_states @ weight.T 得到每个token对所有专家logits
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #一种初始化方法
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        #MoE只关注token值，不关心位置，可以合并batchsize和seq_len维度
        bsz, seq_len, h = hidden_states.shape
        # 将batch、seq_len合并为一维，MoE把每个token当做独立样本，忽略位置信息
        hidden_states = hidden_states.view(-1, h)
        #linear映射，计算出每个token对每个expert的logits
        # gate线性计算 logits: [bsz*seq_len, n_routed_experts]
        logits = F.linear(hidden_states, self.weight, None)

        #步骤1：合并之后对权重进行映射，计算出每个token对各个expert的打分
        #使用softmax进行打分归一化
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        #步骤2： top‑k选出分数最高k个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        #步骤3： top‑k权重归一化 norm_topk_prob
        #一个token分给多个专家，保证权重求和=1，防止数值爆炸
        if self.top_k > 1 and self.norm_topk_prob:
            #token_weight shape:[bsz*seq_len,top_k]
            #对每个token的top_k权重求和->shape:[bsz*seq_len,1]
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20 # + 1e-20防止除以0
            #每个权重除以该token的权重总和，实现归一化
            topk_weight = topk_weight / denominator

        #步骤4：在训练阶段计算负载均衡辅助损失aux_loss
        #负载均衡，防止专家边缘化和专家过度使用
        #两种模式：seq_aux=True序列级 / seq_aux=False batch 全局级

        #seq_aux=True序列级：按每个样本（每条句子）独立计算负载损失；适合 batch 内句子长度差异大场景
        #ce：[bsz, n_routed_experts]统计每条样本内每个专家被选中频次
        #和该样本内专家平均 gate 分数做乘积求和，鼓励每条样本内部专家负载均衡

        #seq_aux=False batch 全局级：整个 batch 所有 token 统计专家选择频率
        # ce：整个 batch 各个专家被选中的平均概率
        # Pi：整个 batch 各个专家 gate 打分平均值
        #aux_loss = (Pi * fi).sum() * self.alpha

        #判断在训练模式，且alpha > 0 （aux_loss*alpha）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores #保存原始分数用于计算辅助损失
            aux_topk = self.top_k
            #将top-k，从[bsz*seq_len,top_k]恢复到[bsz,seq_len,top_k]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            # 方式1，序列级辅助损失（seq_aux=True时）
            #seq_aux bool参数
            if self.seq_aux:
                #恢复scores维度，[bsz*seq_len,n_routed_experts]->[bsz,  seq_len,n_routed_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                #统计每个batch中各个专家被选中的次数
                #ce:[bsz,n_routed_experts]  
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                #scatter_add:根据topk_idx_for_aux_loss的值把1累加到ce中对应位置
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts) #计算得到每个专家的使用率
                # ce * scores_for_seq_aux.mean(dim=1)：专家使用率*该专家的平均分数
                # 鼓励所有专家获得相似的使用率和分数
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha

            # 方式2,batch级辅助损失（seq_aux=False时）
            else:
                #将tok_idx展平，[bsz,seq_len,top_k]->[bsz*seq_len*top_k]
                #转为one-hot编码，得到[bsz*seq_len*top_k,n_routed_experts]
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                # ce：{n_routed_experts},表示整个batch中每个专家的平均选择率
                ce = mask_ce.float().mean(0)
                # Pi：{n_routed_experts},表示整个batch中每个专家的平均分数
                Pi = scores_for_aux.mean(0)
                # fi：标准化的专家选择率（乘以n_routed_experts保持平衡）
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        # topk_idx: [B*S, top_k]：每个 token 选中专家 id
        # topk_weight: [B*S, top_k]，对应专家权重
        # aux_loss：负载均衡损失
        return topk_idx, topk_weight, aux_loss

#替代普通FeedForward：路由专家集合、MoEGate 门控、可选共享专家
class MoEFeedForward(nn.Module):  
    def __init__(self, config: XLLMConfig):
        super().__init__()
        self.config = config
        # 路由专家：n_routed_experts个独立FeedForward（每个专家就是一个标准SwiGLU FFN）
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        # 共享专家，所有token无条件执行，不走gate路由
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 调用门控得到专家索引、权重、辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 展开x以便处理
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)
        if self.training:
        # ---------------- 训练路径 ----------------
        # repeat_interleave：每个token复制top_k份，因为每个token要送给k个专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    # 没有token分配给该专家，保证该专家参数依然参与梯度（防止梯度消失）
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # 最后的y是每个token经过专家处理后的加权结果
            # reshape回 [B*S, top_k, hidden]，乘专家权重，对top‑k维度求和，得到每个token输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)

        else:
            # ----------------推理路径 @torch.no_grad() moe_infer----------------
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        # 加上共享专家输出，共享专家全部token都跑一遍
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    # 训练用repeat_interleave会复制张量，推理会浪费显存；推理采用sort‑batching（打包）优化
    # 1. 将flat_expert_indices排序，把分给同一个专家的 token 连续排在一起
    # 2. bincount统计每个专家分到多少 token；cumsum 得到每个专家 token 分片边界
    # 3. 每个专家一次性批量处理分配给自己的全部 token
    # 4. 输出乘 gate 权重，使用scatter_add_散点加到输出缓存张量对应位置；不做张量复制，节省显存
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache
    


class XLLMBlock(nn.Module):
    def __init__(self,layer_id:int,config:XLLMConfig):
        super().__init__()
        self.num_attention_heads=config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.head_dim=self.hidden_size//self.num_attention_heads
        self.self_attention=Attention(config)

        self.layer_id=layer_id
        self.input_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp=(
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)
        )

    def forward(
            self,
            hidden_states,
            position_embeddings,
            past_key_value=None,
            use_cache=False,
            attention_mask=None,
    ):
        residual=hidden_states
        hidden_states, present_key_value=self.self_attention(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states=residual+hidden_states
        hidden_states=hidden_states+self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states,present_key_value


class XLLMModel(nn.Module):
    def __init__(self,config:XLLMConfig):
        super().__init__()
        self.config=config
        self.vocab_size,self.num_hidden_layers=(
            config.vocab_size,
            config.num_hidden_layers,
        )
        # 词嵌入层：id → hidden_size向量
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)

        self.dropout=nn.Dropout(config.dropout)
        # 堆叠N层Transformer Block XLLMBlock，就是每一层(注意力+FFN)

        self.layers=nn.ModuleList(
            [XLLMBlock(l,config) for l in range(self.num_hidden_layers)]
        )
        # 最后一层输出 RMSNorm 归一化
        self.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        # ========= RoPE / YaRN 预计算旋转位置编码 =========
        # precompute_freqs_cis：预生成复数旋转因子cos、sin，YaRN缩放在这里内部完成
        freqs_cos,freqs_sin=precompute_freqs_cis(
            dim=config.hidden_size//config.num_attention_heads, #单头维度 d_k
            end=config.max_position_embeddings, #最大上下文长度
            rope_base=config.rope_theta, #RoPE base θ，llama是10000
            rope_scaling=config.rope_scaling, #YaRN缩放配置dict
        )
        # register_buffer：注册缓冲区
        #buffer模型参数,不需要梯度、不参与优化器更新；
        # 保存进state_dict；persistent=False：保存checkpoint不存这两个张量，启动重新计算，减少ckpt体积
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self,
            input_ids:Optional[torch.Tensor]=None,
            attention_mask:Optional[torch.Tensor]=None,
            past_key_values:Optional[Tuple[Tuple[torch.Tensor]]]=None,
            use_cache:bool=False,
            **kwargs,
    ):
        batch_size,seq_length=input_ids.shape

        # 兼容huggingface格式，有些包装类past_key_values有.layers属性，做防御，置None
        if hasattr(past_key_values,'layers'):
            past_key_values=None #start_pos=0,第一次输入 prompt,Prefill 阶段
        # 如果没有传入kv cache，生成和层数等长的None列表，每一层对应一个kv缓存
        past_key_values=past_key_values or [None]*len(self.layers)

        # start_pos：解码起始位置。KV Cache增量解码核心
        # 增量生成的时候：已经生成N个token，新token从位置N开始，不需要重新计算前面全部位置的RoPE
        start_pos=(
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # 1. token id 做embedding + dropout
        hidden_states=self.dropout(self.embed_tokens(input_ids))
        # 根据start_pos切片取出当前需要的RoPE cos sin；增量解码只取当前step的位置
        position_embeddings=(
            self.freqs_cos[start_pos:start_pos+seq_length],
            self.freqs_sin[start_pos:start_pos+seq_length],
        )

        presents=[] #收集每一层新输出的 KV cache，use_cache=True才会填充
        # 循环遍历每一层Transformer Block
        for layer_idx,(layer,past_key_value) in enumerate(
            zip(self.layers,past_key_values)
        ):
            # XLLMBlock 前向：输入隐状态、RoPE表、上一步KV缓存、mask
            # 返回：更新后的hidden_states；present=该层新的(key,value)
            hidden_states,present=layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )

            presents.append(present)

        hidden_states=self.norm(hidden_states)

        aux_loss=sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp,MoEFeedForward
                )
             ],
             hidden_states.new_zeros(1).squeeze(),
        )

        # 返回：最后隐状态、全部层的KV缓存
        return hidden_states,presents,aux_loss


class XLLMForCausalLM(PreTrainedModel,GenerationMixin):
    config_class=XLLMConfig

    def __init__(self,config:XLLMConfig):
        self.config=config
        super().__init__(config)
        # Transformer主干
        self.model=XLLMModel(config)
        # lm_head：把hidden_size维度隐状态映射vocab_size词表，得到每个token logits
        self.lm_head=nn.Linear(
            self.config.hidden_size,self.config.vocab_size,bias=False
        )
        #权重共享
        ## 输入embed_tokens 和输出lm_head权重共享。LLaMA系列标准做法，减少参数量，效果稳定
        # embed_tokens.weight: [vocab_size, hidden_size]
        # lm_head.weight: [vocab_size, hidden_size]
        self.model.embed_tokens.weight=self.lm_head.weight

    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                past_key_values:Optional[Tuple[Tuple[torch.Tensor]]]=None,
                use_cache:bool=False,
                logits_to_keep:Union[int,torch.Tensor]=0,
                labels:Optional[torch.Tensor]=None,
                **args
    ):
        # 调用基座XLLMModel，拿到隐状态、KV缓存
        hidden_states,past_key_values,aux_loss=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        # 只取最后N个token计算logits。增量生成的时候，只需要最后1个token预测下一个token
        # logits_to_keep=1 → slice(-1, None)，只拿序列最后一位hidden_state
        # 不对全部seq_len做lm_head矩阵乘法，省显存速度快
        slice_indices=(
            slice(-logits_to_keep,None)
            if isinstance(logits_to_keep,int)
            else logits_to_keep
        )
        logits=self.lm_head(hidden_states[:,slice_indices,:])

        loss=None
        if labels is not None:
            #语言建模：用 logits[..., :-1] 预测 labels[..., 1:]（下一个 token）
            #ignore_index=-100 忽略 PAD 位置，不参与 loss 计算
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss=F.cross_entropy(shift_logits.view(-1,shift_logits.size(-1)),
                                 shift_labels.view(-1),
                                 ignore_index=-100,
            )

        output=CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss=aux_loss
        return  output
#(1) Dense Model
# input_ids (B, L)
#     ↓
# XLLMModel.embed_tokens → embedding + dropout
#     ↓
# for XLLMBlock 每一层：RoPE旋转 → MHA注意力 → RMSNorm → FFN
#     ↓
# RMSNorm → hidden_states(B,L,H)
#     ↓
# XLLMForCausalLM.lm_head（权重共享）
#     ↓
# logits (B, keep_len, vocab_size) → 计算loss / token采样生成



#(2) MoE
# hidden_states [B,S,H]
#         ↓
# MoEGate：计算每个token对各专家gate分数 → top‑k专家索引、权重、aux_loss
#         ↓
# if train:
#     token重复复制top_k次，分发到各个专家FFN，专家输出乘权重求和
# else infer:
#     token按所属专家排序打包，每个专家批量处理分配给自己的token，scatter_add聚合输出
#         ↓
# 可选：加上共享专家全部token的FFN输出
#         ↓
# MoEFeedForward输出 [B,S,H]，残差加到主链路

