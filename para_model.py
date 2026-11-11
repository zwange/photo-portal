"""ArtiMuse numeric scorer backed by the official inference source."""
import ast,json,os,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"vendor"/"ArtiMuse"/"src"/"artimuse"
sys.path.insert(0,str(SOURCE))

def load_scorer(variant):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from transformers import AutoTokenizer
    from internvl.conversation import get_conv_template
    from internvl.model.internvl_chat.aes_tokens import AESTHETICS_TOKEN_LIST
    from internvl.model.internvl_chat.modeling_artimuse import InternVLChatModel

    torch.set_num_threads(4)
    torch.manual_seed(0)
    checkpoint = ROOT / f"checkpoint_{variant}"
    revision = json.loads((ROOT / f"{variant}_revision.json").read_text())
    print("Loading", revision, flush=True)
    load_start = time.perf_counter()
    model = InternVLChatModel.from_pretrained(
        str(checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=False, local_files_only=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=False, local_files_only=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded {parameter_count:,} parameters in {load_seconds:.2f}s", flush=True)
    source = ast.parse((SOURCE / "internvl/model/internvl_chat/modeling_artimuse.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "InternVLChatModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "score")
    question = next(ast.literal_eval(n.value) for n in fn.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "question" for t in n.targets))
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], "<image>\n" + question)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt().replace("<image>", "<img>" + "<IMG_CONTEXT>"*model.num_image_token + "</img>", 1)
    tokens = tokenizer(prompt, return_tensors="pt").to("cuda")
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    preferred = torch.tensor([tokenizer.convert_tokens_to_ids(w) for w in AESTHETICS_TOKEN_LIST], device="cuda")
    assert len(set(preferred.tolist())) == 101 and tokenizer.unk_token_id not in preferred.tolist()
    weights = torch.arange(101, device="cuda", dtype=torch.float32)
    transform = T.Compose([T.Resize((448,448), interpolation=T.InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])

    def pixels(row):
        with Image.open(row["thumbnail_path"]) as im:
            return transform(im.convert("RGB"))

    @torch.inference_mode()
    def score(batch):
        n = len(batch)
        output = model.generate_logits(pixel_values=batch, input_ids=tokens.input_ids.repeat(n,1),
                                       attention_mask=tokens.attention_mask.repeat(n,1))
        return (torch.softmax(output.logits[:,-1,preferred].float(),dim=-1) @ weights).cpu().tolist()

    return torch, pixels, score, model, tokenizer, revision, load_seconds, parameter_count
