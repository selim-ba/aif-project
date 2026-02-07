# part 4 - app/rag/rag_model.py

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPProcessor, CLIPModel

class RAG:
    def __init__(self, config, index, id_map, brochure):
        self.index = index
        self.id_map = id_map
        self.brochure = brochure
        self.config = config
        
        # Setup hardware
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu" # pb avec mps sur mac, on force l'usage du cpu pour l'inference
        print(f" RAG Inference running on: {self.device.upper()}")

        # LLM (Qwen)
        print(" Loading Qwen...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config["FOUND_MODEL_PATH"])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                config["FOUND_MODEL_PATH"], 
                #torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise e

        # CLIP
        print(f"Loading CLIP on {self.device}...")
        try:
            self.clip_model = CLIPModel.from_pretrained(config["CLIP_MODEL_ID"]).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(config["CLIP_MODEL_ID"])
            self.clip_model.eval()
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            raise e

        self.chat_history = []

    def retrieve(self, query, n_results=5):
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                text_features = outputs
            elif hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            elif hasattr(outputs, 'text_embeds'):
                text_features = outputs.text_embeds
            else:
                text_features = outputs[0]
            
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            query_vector = text_features.cpu().detach().numpy()[0]

        try:
            indices, distances = self.index.get_nns_by_vector(query_vector, n_results, include_distances=True)
        except Exception as e:
            print(f"Annoy Index Error: {e}")
            return []

        results = []
        for idx, dist in zip(indices, distances):
            # raw_key = self.id_map.get(idx)
            raw_key = self.id_map.get(str(idx))
            brochure_key = raw_key
            if isinstance(raw_key, dict):
                brochure_key = list(raw_key.values())[0]

            if brochure_key in self.brochure:
                results.append(self.brochure[brochure_key])
            elif str(brochure_key) in self.brochure:
                results.append(self.brochure[str(brochure_key)])
            elif isinstance(brochure_key, str) and brochure_key.isdigit() and int(brochure_key) in self.brochure:
                results.append(self.brochure[int(brochure_key)])
        return results

    def ask(self, query):
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return "I couldn't find any movies matching your description."

        # Construction du contexte 
        context_pieces = []
        for doc in retrieved_docs:
            plot = doc.get('plot') or doc.get('movie_plot') or ""
            genre = doc.get('category') or doc.get('movie_category') or "Unknown"
            ref = doc.get('poster_path') or str(doc.get('id', 'Unknown'))
            
            context_pieces.append(f"--- CANDIDATE MOVIE ---\n[Image Ref: {ref}]\nGenre: {genre}\nPlot: {plot[:600]}")
        
        context_text = "\n".join(context_pieces)

        system_prompt = self.config["SYSTEM_PROMPT"]
        
        # User prompt + contexte
        user_prompt = f"""You are a movie database assistant.
User Request: "{query}"

Here are the best matches found (WARNING: Titles are missing from the database, but they may be extracted from plots):
{context_text}

INSTRUCTIONS:
1. Select the best movie based on the plot and try to come up with a title.
2. Be honest: State clearly that you don't have the exact title.
3. Describe the plot and why it fits.

### EXAMPLE OF ANSWER FORMAT:
"Since the titles are missing in my database, I cannot give you the exact name, but based on the plot, it looks like [Title Guess].
This movie fits your request because..."

Assistant:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # with torch.no_grad():
        #     generated_ids = self.model.generate(
        #         model_inputs.input_ids,
        #         attention_mask=model_inputs.attention_mask,
        #         max_new_tokens=400, 
        #         temperature=0.5, # Température moyenne : on veut qu'il ose deviner un peu
        #         do_sample=True,
        #         top_k=50,
        #         top_p=0.95,
        #         pad_token_id=self.tokenizer.eos_token_id
        #     )
        
        # input_length = model_inputs.input_ids.shape[1]
        # new_tokens = generated_ids[:, input_length:]
        # raw_response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        

        model_inputs = self.tokenizer([text], return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=400,
                temperature=0.5,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = model_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]
        raw_response = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        # --- NETTOYAGE ---
        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        if '<think>' in clean_response:
             clean_response = clean_response.split('<think>')[0]
        
        clean_response = clean_response.strip()

        # Fallback
        if not clean_response:
             clean_response = "I found a matching movie but I cannot determine the title."

        self.chat_history.append((query, clean_response))
        return clean_response

    def reset_chat(self):
        self.chat_history = []