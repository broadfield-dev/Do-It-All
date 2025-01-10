import gradio as gr
from doitall.module.gradio_theme import theme
import random
#from main import Do_It_All
import doitall.module.rag_main as main

def isV(inp,is_=False):  # Verbose
    if is_==True:
        print(inp)
        is_=False

clients_main=[]
client_hf=[
    {'type':'image','loc':'hf','key':'','name':'black-forest-labs/FLUX.1-dev','rank':'op','max_tokens':16384,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'deepseek-ai/DeepSeek-V2.5-1210','rank':'op','max_tokens':16384,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'Qwen/Qwen2.5-Coder-32B-Instruct','rank':'op','max_tokens':32768,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'katanemo/Arch-Function-3B','rank':'op','max_tokens':16384,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'PowerInfer/SmallThinker-3B-Preview','rank':'op','max_tokens':32768,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'Qwen/QwQ-32B-Preview','rank':'op','max_tokens':16384,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'Qwen/QVQ-72B-Preview','rank':'op','max_tokens':32768,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'meta-llama/Llama-3.2-1B','rank':'op','max_tokens':32768,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'Snowflake/snowflake-arctic-embed-l-v2.0','rank':'op','max_tokens':4096,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'Snowflake/snowflake-arctic-embed-m-v2.0','rank':'op','max_tokens':4096,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'HuggingFaceTB/SmolLM2-1.7B-Instruct','rank':'op','max_tokens':40000,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
    {'type':'text','loc':'hf','key':'','name':'mistralai/Mixtral-8x7B-Instruct-v0.1','rank':'op','max_tokens':40000,'schema':{'bos':'<s>','eos':'</s>'}},
]

client_gradio=[
    {'type':'text','loc':'gradio','key':'','name':'bigscience/bloomz','rank':'op','max_tokens':16384,'schema':{'bos':'<|im_start|>','eos':'<|im_end|>'}},
]

client_ollama=[
    {'type':'text','loc':'ollama','key':'','name':'tinyllama','rank':'op','max_tokens':16384,'schema':{'bos':['<|system|>','<|user|>','<|assistant|>'],'eos':'<|im_end|>'}},
]
clients_main.extend(client_hf)
clients_main.extend(client_gradio)
clients_main.extend(client_ollama)
clients_out=clients_main
head="""
      <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
          <script type = 'text/javascript'>
            mermaid.initialize({ startOnLoad: true });
            </script>
      """ 
js_="""
    function (inp) {
    // Get the button and graph container
    // const button = document.getElementById('test_box');
    const graphDiv = document.getElementById('test_merm');
    console.log('inp')
    console.log(inp)
    console.log(graphDiv)
    // Function to update the Mermaid graph

         // Create a new div element for the graph
        const graphElement = document.createElement('div');
        graphElement.className = 'mermaid';
        graphElement.textContent = inp;

        // Clear the previous graph and add the new one
        graphDiv.innerHTML = '';
        graphDiv.appendChild(graphElement);

        // Reinitialize Mermaid to render the new graph
        mermaid.run();
    }"""        
do_it=main.Do_It_All(clients=clients_out)
def load_merm(inp):
    if inp != None:
        outp=do_it.merm_html.replace('CONTENT',inp)
        return outp
    else:
        pass
css="""
#prompt_box textarea{
    color:white;
  }
  
  span.svelte-5ncdh7{
  color:white;}"""
def check_ch(inp,inp_val):
    if inp == True and inp_val <= 1:
        value = random.randint(1,9999999999999)
    elif inp == True and inp_val > 1:
        value = do_it.seed_val
    else:
        value=inp_val
    return value
def check_box():
    return True
def main():
    with gr.Blocks(head=head,theme=theme,css=css) as ux:
        gr.HTML("""<center><div style='font-family:monospace;font-size:xxx-large;font-weight:900;'>Do-it-All</div><br>
                <div style='font-size:large;font-weight:700;'>Basic AI Agent System</div><br>
                <div style='font-size:large;font-weight:900;'></div><br>
                """)
    
        with gr.Row():
            
            with gr.Column(scale=1):
                m_html=gr.HTML("<div id='test_merm'></div>")
                h_html=gr.HTML("<div id='test_merm'></div>")

            with gr.Column(scale=3):


                prompt=gr.MultimodalTextbox(label="Prompt", elem_id="prompt_box", file_count="multiple", file_types=["image",'.pdf','.txt','.html','.json','.css','.js','.py','.svg'])
                chatbot2=gr.Chatbot(label="Thoughts",type='messages',show_label=False,height=200,max_height=200, show_share_button=False, show_copy_button=False, layout="panel")
                chatbot=gr.Chatbot(label="Chatbot",type='messages',show_label=False, height=800, show_share_button=False, show_copy_button=True, layout="panel")

            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column():
                        seed_ch=gr.Checkbox(label="Random",value=False)
                        seed=gr.Number(label="Seed",step=1,precision=0,value=do_it.seed_val,interactive=True)
                    with gr.Column():
                        mod_c=gr.Dropdown(label="Model",choices=[n['name'] for n in do_it.clients],value='Qwen/Qwen2.5-Coder-32B-Instruct',type='index')
                        tok_in=gr.Textbox(label='HF TOKEN')
                with gr.Row():
                    submit_b = gr.Button()
                    stop_b = gr.Button("Stop")
                    clear = gr.ClearButton([chatbot,prompt])

                text_trig=gr.Textbox(elem_id='test_box')
        ux.load(check_box,None,seed_ch)
        seed_ch.change(check_ch,[seed_ch,seed],seed)
        sub_b = submit_b.click(check_ch,[seed_ch,seed],seed).then(do_it.agent, [prompt,chatbot,mod_c,tok_in,seed_ch,seed],[chatbot,text_trig,h_html,chatbot2])
        sub_p = prompt.submit(check_ch,[seed_ch,seed],seed).then(do_it.agent, [prompt,chatbot,mod_c,tok_in,seed_ch,seed],[chatbot,text_trig,h_html,chatbot2])
        chatbot2.change(check_ch,[seed_ch,seed],seed)
        text_trig.change(load_merm,text_trig,None,js=js_)
        stop_b.click(None,None,None, cancels=[sub_b,sub_p])
    ux.queue(default_concurrency_limit=20).launch(max_threads=40)
if __name__ == '__main__':
    main()
