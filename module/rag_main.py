from huggingface_hub import InferenceClient
#from text_generation import InferenceAPIClient
#import gradio as gr
import ollama
#import openai
import datetime
import requests
import random
import prompts1 as prompts
import uuid
import json
import bs4
import os
import lxml
from langchain.embeddings import OpenAIEmbeddings 
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveJsonSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from pypdf import PdfReader

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings,ChatHuggingFace

class RagClient:
    def __init__(self):
        try:
            self.clientDir = "./dbbbb/"
            print(self.clientDir)
            if not os.path.isdir(self.clientDir):os.mkdir(self.clientDir)
            #self.client = chromadb.CloudClient()
            self.collection_name = 'memory'
        
            #from sentence_transformers import SentenceTransformer
            #self.model_name = "all-MiniLM-L6-v2"
            self.model_list= ['sentence-transformers/all-mpnet-base-v2',
                              'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
                              'sentence-transformers/roberta-base-nli-mean-tokens',
                              'dunzhang/stella_en_400M_v5',
                              'shibing624/text2vec-base-chinese',
                              'sentence-transformers/stsb-roberta-base',
                              'dunzhang/stella_en_1.5B_v5',
                              'Alibaba-NLP/gte-multilingual-base',
                              ]
            self.model_name = self.model_list[0]
            #sentence_transformer_model = SentenceTransformer(model_name)

            # Create LangChain embeddings
        except Exception as e:
            print(e)
    def read_pdf(pdf_path):
        if not os.path.isdir("./images"):
            os.mkdir("./images")
        text=[]
        images = ""
        reader = PdfReader(f'{pdf_path}')
        number_of_pages = len(reader.pages)
        file_name=str(pdf_path).split("\\")[-1]
        for i in range(number_of_pages):
            page = reader.pages[i]
            images=""
            if len(page.images) >0:
                for count, image_file_object in enumerate(page.images):
                    with open( "./images/" + str(count) + image_file_object.name, "wb") as fp:
                        fp.write(image_file_object.data)
                    #buffer = io.BytesIO(image_file_object.data)
                    #images.append({"name":file_name,"page":i,"cnt":count,"image":Image.open(buffer)})
                    #images.append(str(image_file_object.data))
                    images += "./images/" + str(count) + image_file_object.name + "\n"
                #text = f'{text}\n{page.extract_text()}'
            else:
                images=""
            text.append({"page":i,"text":page.extract_text(),"images":images})
        return text
    def save_memory(self,file_in):
        print('save memory')
        try:
            client = Chroma()

            if str(file_in).endswith(('.txt','.html','.json','.css','.js','.py','.svg')):
                with open(str(file_in), "r") as file:
                    document_text = file.read()
                file.close()
            elif str(file_in).endswith(('.pdf')):
                   pdf_json=self.read_pdf(str(file_in))
                   document_text='PDF_CONTENT: ' +json.dumps(pdf_json,indent=1)
            else: document_text = str(file_in)
            # Split the document into chunks
            if type(document_text)==type({}):
                text_splitter = RecursiveJsonSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_json(document_text)
                print('recursive json split: ')
            else:    
                text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_text(document_text)
                print('character text split: ')

            ids=[]
            for x in texts:
                ids.append(str(uuid.uuid4()))
            print(len(ids))
            meta_d=[{'timestamp':str(datetime.datetime.now()),'filename':file_in}]
            client.from_texts(texts,HuggingFaceEmbeddings(model_name=self.model_name),metadatas=[]+meta_d,persist_directory=self.clientDir)
            print('loaded')

            #print(self.collection)
        except Exception as e:
            print(e)
    def recall_memory(self, query_in):
        print('recall memory')
        print('QUERY IN: ', query_in)
        embeddings=HuggingFaceEmbeddings(model_name=self.model_name)
        embeded=embeddings.embed_query(str(query_in))
        db2 = Chroma(persist_directory=self.clientDir, embedding_function=HuggingFaceEmbeddings(model_name=self.model_name))
        print("Loaded")
        MAX_TOKENS=20000
        #db2.similarity_search_by_vector
        results = db2.similarity_search_by_vector(embeded, k=int(10))
        return results
def isV(inp,is_=False):  # Verbose
    if is_==True:
        print(inp)
        is_=False

class Do_It_All:
    def __init__(self,clients):
        self.MAX_HISTORY=15000
        self.save_settings=[{}]
        self.merm_html="""CONTENT"""    
        self.html_html="""<iframe 
        src="data:text/html,<p>**CONTENT**</p>" 
        width="400" 
        height="200"
        sandbox="allow-scripts"
        ></iframe>"""
        self.seed_val=1
        self.clients = clients
        self.roles = []


    def gen_im(self,prompt,seed):
        isV('generating image', True)
        im_client=InferenceClient(self.clients[0]['name'])
        image_out = im_client.text_to_image(prompt=prompt,height=256,width=256,num_inference_steps=10,seed=seed)
        output=f'{uuid.uuid4()}.png'
        image_out.save(output)
        isV(('Done: ', output), True)
        return [{'role':'assistant','content': {'path':output}}]

    def compress_data(self,c,purpose, history,mod,tok,seed,data):
        isV(data)
        resp=[None,]
        #if not data: data=[];data[0]="NONE"
        seed=random.randint(1,1000000000)
        isV (c)
        divr=int(c)/self.MAX_HISTORY
        divi=int(divr)+1 if divr != int(divr) else int(divr)
        chunk = int(int(c)/divr)
        isV(f'chunk:: {chunk}')
        isV(f'divr:: {divr}')
        isV (f'divi:: {divi}')
        task1="refine this data"
        out = []
        #out=""
        s=0
        e=chunk
        isV(f'e:: {e}')
        new_history=""
        task = f'Compile this data to fulfill the task: {task1}, and complete the purpose: {purpose}\n'
        for z in range(divi):
            data[0]=new_history
            isV(f's:e :: {s}:{e}')
            hist = history[s:e]
            resp = self.generate(
                prompt=purpose,
                history=hist,
                mod=int(mod),
                tok=int(tok),
                seed=int(seed),
                role='COMPRESS',
                data=data,
            )
            resp_o = list(resp)[0]
            new_history = resp_o
            isV (resp)
            out+=resp_o
            e=e+chunk
            s=s+chunk
        
        isV ("final" + resp_o)
        #history = [{'role':'system','content':'Compressed History: ' + str(resp_o)}]
        return str(resp_o)

    def find_all(self,prompt,history, url,mod,tok,seed,data):
        return_list=[]
        isV (f"trying URL:: {url}")        
        try:
            if url != "" and url != None:    
                out = []
                source = requests.get(url)
                isV('status: ', source.status_code)
                if source.status_code ==200:
                    soup = bs4.BeautifulSoup(source.content,'lxml')
                    rawp=(f'RAW TEXT RETURNED: {soup.text}')
                    cnt=0
                    cnt+=len(rawp)
                    out.append(rawp)
                    out.append("HTML fragments: ")
                    q=("a","p","span","content","article")
                    for p in soup.find_all("a"):
                        out.append([{"LINK TITLE":p.get('title'),"URL":p.get('href'),"STRING":p.string}])
                    c=0
                    out2 = str(out)

                    if len(out2) > self.MAX_HISTORY:
                        isV("compressing...")
                        rawp = self.compress_data(len(out2),prompt,out2,mod,tok,seed, data)  
                    else:
                        isV(out)
                        rawp = out
                    return rawp
                else:
                    history.extend([{'role':'system','content':f"observation: That URL string returned an error: {source.status_code}, I should try a different URL string\n"}])
                    
                    return history
            else: 
                history.extend([{'role':'system','content':"observation: An Error occured\nI need to trigger a search using the following syntax:\naction: INTERNET_SEARCH action_input=URL\n"}])
                return history
        except Exception as e:
            isV (e)
            history.extend([{'role':'system','content':"observation: I need to trigger a search using the following syntax:\naction: INTERNET_SEARCH action_input=URL\n"}])
            return history

        
    def format_prompt(self,message, mod, system):
        eos=f"{self.clients[int(mod)]['schema']['eos']}\n"
        bos=f"{self.clients[int(mod)]['schema']['bos']}\n"
        prompt=""
        prompt+=bos
        prompt+=system
        prompt+=eos
        prompt+=bos
        prompt += message
        prompt+=eos
        prompt+=bos
        return prompt
    def llama_load():
        #llm = Llama.from_pretrained(
        #    repo_id="bullerwins/DeepSeek-V3-GGUF",
        #    filename="DeepSeek-V3-GGUF-bf16/DeepSeek-V3-Bf16-256x20B-BF16-00001-of-00035.gguf",
        # )
        pass
                
    def generate(self,prompt,history,mod=2,tok=4000,seed=1,role="RESPOND",data=None):
        isV(role)
        current_time=str(datetime.datetime.now())
        timeline=str(data[4])
        self.roles=[{'name':'MANAGER','system_prompt':str(prompts.MANAGER.replace("**CURRENT_TIME**",current_time).replace("**TIMELINE**",timeline).replace("**HISTORY**",str(history)))},
                {'name':'PATHMAKER','system_prompt':str(prompts.PATH_MAKER.replace('**STEPS**',str(data[2])).replace("**CURRENT_OR_NONE**",timeline).replace("**PROMPT**",json.dumps(data[0],indent=4)).replace("**HISTORY**",str(history)))},
                {'name':'INTERNET_SEARCH','system_prompt':str(prompts.INTERNET_SEARCH.replace("**TASK**",str(prompt)).replace("**KNOWLEDGE**",str(data[3])).replace("**HISTORY**",str(history)))},
                {'name':'COMPRESS','system_prompt':str(prompts.COMPRESS.replace("**TASK**",str(prompt)).replace("**KNOWLEDGE**",str(data[0])).replace("**HISTORY**",str(history)))},
                {'name':'RESPOND','system_prompt':str(prompts.RESPOND.replace("**CURRENT_TIME**",current_time).replace("**PROMPT**",prompt).replace("**HISTORY**",str(history)).replace("**TIMELINE**",timeline))},
                ]
        roles = self.roles
        g=True
        for roless in roles:
            if g==True:
                if roless['name'] == role:
                    system_prompt=roless['system_prompt']
                    isV(system_prompt)
                    g=False
                else: system_prompt = ""
                    
        formatted_prompt = self.format_prompt(prompt, mod, system_prompt)
        
        if tok==None:isV('Error: tok value is None')
        isV("tok",tok)
        self.generate_kwargs = dict(
            temperature=0.99,
            max_new_tokens=tok, #total tokens - input tokens
            top_p=0.99,
            repetition_penalty=1.0,
            do_sample=True,
            seed=seed,
        )
        output = ""
       
        if self.clients[int(mod)]['loc'] == 'hf':
            isV("Running ", self.clients[int(mod)]['name'])
            client=InferenceClient(self.clients[int(mod)]['name'])
            stream = client.text_generation(formatted_prompt, **self.generate_kwargs, stream=True, details=True, return_full_text=True)
            for response in stream:
                output += response.token.text
            yield output.replace('<|im_start|>','').replace('<|im_end|>','')
            yield history
            yield prompt

        elif self.clients[int(mod)]['loc'] == 'ollama':
            isV("Running ", role)
            client=InferenceClient(self.clients[int(mod)]['name'])
            stream = client.text_generation(formatted_prompt, **self.generate_kwargs, stream=True, details=True, return_full_text=True)
            for response in stream:
                output += response.token.text
            yield output.replace('<|im_start|>','').replace('<|im_end|>','')
            yield history
            yield prompt

        '''elif self.clients[int(mod)]['loc'] == 'gradio':
            isV("Running ", role)

            client = InferenceAPIClient(self.clients[int(mod)]['name'])
            for response in client.generate_stream(formatted_prompt, **self.generate_kwargs, return_full_text=True):
                if not response.token.special:
                    output += response.token.text

            yield output.replace('<|im_start|>','').replace('<|im_end|>','')
            yield history
            yield prompt'''


    
    def multi_parse(self,inp):
        parse_boxes=[
            {'name':'json','cnt':7},
            {'name':'html','cnt':7},
            {'name':'css','cnt':6},
            {'name':'mermaid','cnt':10},
            ]
        isV("PARSE INPUT")
        isV(inp)
        if type(inp)==type(""):
            lines=""
            if "```" in inp:
                gg=True
                for ea in parse_boxes:
                    if gg==True:
                        if f"""```{ea['name']}""" in inp:
                            isV(f"Found {ea['name']} Code Block")
                            start = inp.find(f"```{ea['name']}") + int(ea['cnt'])  
                            end = inp.find("```", start) 
                            if start >= 0 and end >= 0:
                                inp= inp[start:end] 
                            else:
                                inp="NONE" 
                            isV("Extracted Lines")
                            isV(inp)
                            gg=False
                            return {'type':f"{ea['name']}",'string':str(inp)}
                        
            else:isV('ERROR: Code Block not detected')
        else:isV("ERROR: self.multi_parse requires a string input")

    def parse_from_str(self,inp):   
        rt=True
        out_parse={}
        for line in inp.split("\n"):
            if rt==True:
                if "```" in line:
                    out_parse=self.multi_parse(inp)
                    #rt=False
        if out_parse and out_parse['type']=='html':
            isV('HTML code: TRUE')
            html=self.html_html.replace('**CONTENT**',out_parse['string'].replace(","," "))               
            #parse_url='https://'
        else:
            html=""
            isV('HTML code: TRUE')
        return html


    def parse_file_json(self,inp):
        isV("PARSE INPUT")
        isV(inp)
        if type(inp)==type(""):
            lines=""
            if "```json" in inp:
                start = inp.find("```json") + 7  
                end = inp.find("```", start) 
                if start >= 0 and end >= 0:
                    inp= inp[start:end] 
                else:
                    inp="NONE" 
                isV("Extracted Lines")
                isV(inp)
            try:
                out_json=eval(inp)
                out1=str(out_json['filename'])
                out2=str(out_json['filecontent'])
                return out1,out2
            except Exception as e:
                isV(e)
                return "None","None"
        if type(inp)==type({}):
            out1=str(inp['filename'])
            out2=str(inp['filecontent'])
            return out1,out2
    def read_pdf(self,pdf_path):
        if not os.path.isdir("./images"):
            os.mkdir("./images")
        text=[]
        images = ""
        reader = PdfReader(f'{pdf_path}')
        number_of_pages = len(reader.pages)
        file_name=str(pdf_path).split("\\")[-1]
        for i in range(number_of_pages):
            page = reader.pages[i]
            images=""
            if len(page.images) >0:
                for count, image_file_object in enumerate(page.images):
                    with open( "./images/" + str(count) + image_file_object.name, "wb") as fp:
                        fp.write(image_file_object.data)
                    #buffer = io.BytesIO(image_file_object.data)
                    #images.append({"name":file_name,"page":i,"cnt":count,"image":Image.open(buffer)})
                    #images.append(str(image_file_object.data))
                    images += "./images/" + str(count) + image_file_object.name + "\n"
                #text = f'{text}\n{page.extract_text()}'
            else:
                images=""
            text.append({"page":i,"text":page.extract_text(),"images":images})
        return text
               
    def agent(self,prompt_in,history,mod=2,tok_in="",rand_seed=True,seed=1,max_thought=5):
        thought_hist=[{'role':'assistant','content':'Starting'}]
        if not history:history=[]
        history.extend([{'role':'user','content':prompt_in['text']}])
        out_hist=history.copy()
        thought_hist.extend([{'role':'assistant','content':'Starting RAG system...'}])
        merm="graph TD;A[Thought path...];"
        html=""
        yield out_hist,merm,html,thought_hist
        rag=RagClient()

        isV(prompt_in,True)
        isV(('mod ',mod),True)
        in_data=["None","None","None","None","None","Mems",]
        #in_data[0]=prompt_in['text']
        prompt=prompt_in['text']
        fn=""
        com=""
        go=True
        MAX_DATA=int(self.clients[int(mod)]['max_tokens'])*2
 
        cnt=max_thought
        in_data[0]=prompt_in
        file_box=[]
        for file_in in prompt_in['files']:
            thought_hist.extend([{'role':'assistant','content':'Loading user file...'}])

            yield out_hist,merm,html,thought_hist

            if str(file_in).endswith(('.txt','.html','.json','.css','.js','.py','.svg')):
                    rag.save_memory(file_in=file_in)
                    with open(str(file_in), "r") as file:
                        file_box.extend([{'doc_name':file_in,'content':file.read()}])
                    file.close()
            elif str(file_in).endswith(('.pdf')):
                   pdf_json=self.read_pdf(str(file_in))
                   file_box.extend([{'doc_name':file_in,'content':'PDF_CONTENT'+json.dumps(pdf_json,indent=1)}])    
        if len(str(file_box)) > self.MAX_HISTORY:
            file_out = self.compress_data(len(str(file_box)),prompt, str(file_box),mod,10000,seed,in_data)
        else: file_out=str(file_box)
        history.extend([{'role':'assistant','content':"File Content: " + str(file_out)}])
        thought_hist.extend([{'role':'assistant','content':'Starting main script...'}])

        yield out_hist,merm,html,thought_hist
        while go == True:
            mems=rag.recall_memory(history[-1:])
            in_data[5]=mems
            history.extend([{'role':'assistant','content':"Memories: " + str(mems)}])

            print(mems)
            if max_thought==0:
                in_data[2]="Unlimited"
            else:
                in_data[2]=cnt
            #isV(history)
            if rand_seed==True:
                seed = random.randint(1,99999999999999)
            else:
                seed = seed
            self.seed_val=seed
            c=0
            if len(str(history)) > self.MAX_HISTORY:
                #history = [{'role':'assistant','content':self.compress_data(len(str(history)),prompt,history,mod,2400,seed, in_data)  }]
                history = ([{'role':'user','content':str(prompt_in)},{'role':'assistant','content':history[-2:] if len(str(history[-2:])) > self.MAX_HISTORY else history[-1:]}])
            isV('history',False)
            isV('calling PATHMAKER')
            role="PATHMAKER"
            #in_data[3]=file_list
            
            thought_hist.extend([{'role':'assistant','content':'Making Plan...'}])
            yield out_hist,merm,html,thought_hist
            outph=self.generate(prompt,history,mod,2400,seed,role,data=in_data)
            path_out = str(list(outph)[0])
            #history.extend([{'role':'system','content':path_out}])
            out_parse={}
            rt=True
            for line in path_out.split("\n"):
                if rt==True:
                    if "```" in line:
                        out_parse=self.multi_parse(path_out)
                        #rt=False
            if out_parse and out_parse['type']=='mermaid':
                isV('Mermaid code: TRUE')
                merm=self.merm_html.replace('CONTENT',out_parse['string'].replace(","," "))
            elif out_parse and out_parse['type']=='html':
                isV('HTML code: TRUE')
                #html=self.html_html.replace('CONTENT',out_parse['string'].replace(","," "))               
            else:
                html=""
                isV('HTML code: TRUE')
                #html=self.html_html.replace('CONTENT',out_parse['string'].replace(","," "))          
            thought_hist.extend([{'role':'assistant','content':'Choosing Path...'}])
            yield out_hist,merm,html,thought_hist
            
            in_data[4]=str(merm)
            isV("HISTORY: ",history)
            isV('calling MANAGER')
            role="MANAGER"
            outp=self.generate(prompt,history,mod,128,seed,role,in_data)
            outp0=list(outp)[0].split('<|im_end|>')[0]
            isV("Manager: ", outp0)
            #outp0 = re.sub('[^a-zA-Z0-9\s.,?!%()]', '', outpp)
            history.extend([{'role':'assistant','content':str(outp0)}])
            #yield history
            for line in outp0.split("\n"):
                if "action:" in line:
                    try:
                        com_line = line.split('action:')[1]
                        fn = com_line.split('action_input=')[0]
                        com = com_line.split('action_input=')[1].split('<|im_end|>')[0]
                        isV(com)
                        thought_hist.extend([{'role':'assistant','content':f'Calling command: {fn}'}])
                        thought_hist.extend([{'role':'assistant','content':f'Command input: {com}'}])
                        yield out_hist,merm,html,thought_hist
                    except Exception as e:
                        pass
                        fn="NONE"
                    
                    if 'RESPOND' in fn:
                        isV("RESPOND called")
                        in_data[1]=com
                        thought_hist.extend([{'role':'assistant','content':'Formulating Response...'}])
                        yield out_hist,merm,html,thought_hist
                        ret = self.generate(prompt, history,mod,10000,seed,role='RESPOND',data=in_data)
                        ret_out=str(list(ret)[0])
                        yield from ret_out
                        hist_catch=[{'role':'assistant','content':ret_out}]
                        out_hist.extend(hist_catch)
                        rag.save_memory(ret_out)
                        history.extend(hist_catch)
                        history.extend([{'role':'assistant','content':'All tasks are complete, call: COMPLETE'}])
                        yield out_hist,merm,html,thought_hist

                    elif 'IMAGE' in fn:
                        thought_hist.extend([{'role':'assistant','content':'Generating Image...'}])
                        yield out_hist,merm,html,thought_hist
                        isV('IMAGE called',True)
                        out_im=self.gen_im(prompt,seed)
                        out_hist.extend(out_im)
                        yield out_hist,merm,html,thought_hist
    
                    elif 'INTERNET_SEARCH' in fn:
                        thought_hist.extend([{'role':'assistant','content':'Researching Topic...'}])
                        yield out_hist,merm,html,thought_hist
                        isV('INTERNET_SEARCH called',True)
                        ret = self.find_all(prompt, history, com,mod,10000,seed,data=in_data)
                        in_data[3]=str(ret)
                        thought_hist.extend([{'role':'assistant','content':'Compiling Report...'}])
                        yield out_hist,merm,html,thought_hist
                        res_out = self.generate(prompt, history,mod,10000,seed,role='INTERNET_SEARCH',data=in_data)
                        res0=str(list(res_out)[0])
                        rag.save_memory(res0)
                        #html = self.parse_from_str(res0)
                        history.extend([{'role':'system','content':f'RETURNED SEARCH CONTENT: {str(res0)}'}])
                        history.extend([{'role':'system','content':'thought: I have responded with a report of my recent internet search, this step is COMPLETE'}])
                        out_hist.extend([{'role':'assistant','content':f'{str(res0)}'}])
                        yield out_hist,merm,html,thought_hist
                    
                    elif 'COMPLETE' in fn:
                        isV('COMPLETE',True)
                        history.extend([{'role':'system','content':'Complete'}])
                        thought_hist.extend([{'role':'assistant','content':'Complete'}])
                        yield out_hist,merm,html,thought_hist
                        go=False
                        break
                    elif 'NONE' in fn:
                        isV('ERROR ACTION NOT FOUND',True)
                        history.extend([{'role':'system','content':f'observation:The last thing we attempted resulted in an error, check formatting on the tool call'}])
                    else:
                        history.extend([{'role':'system','content':'observation: The last thing I tried resulted in an error, I should try selecting a different available tool using the format, action:TOOL_NAME action_input=required info'}])
                        pass;seed = random.randint(1,9999999999999)
            
            if max_thought > 0:
                cnt-=1
                if cnt <= 0:
                    thought_hist.extend([{'role':'assistant','content':f'observation: We have used more than the Max Thought Limit, ending chat'}])
                    yield out_hist,merm,html,thought_hist
                    go=False
                    break
                
