import streamlit as st
import openai
import boto3
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gc
import tiktoken

access_key = st.secrets["access_key"]
secret_key = st.secrets["secret_key"]
openai.api_key = st.secrets["openai_key"]
gpt_model = "gpt-4"
token_counter = tiktoken.encoding_for_model(gpt_model)
MAX_TOKEN_MODEL = 8000


def num_tokens_from_string(text):
    return len(token_counter.encode(text))


def get_run_gpt_prompt(prompt, in_text, model='gpt-4'):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt}
            , {"role": "user", "content": in_text}
        ],
        max_tokens = MAX_TOKEN_MODEL - num_tokens_from_string(prompt + "\n" + in_text) - 150,
        temperature=0.1
    )
    return response["choices"][0]["message"]["content"]    


def run_prompt_on_all_chunks(prompt, in_texts, model='gpt-4'):
    chunks = []
    for i in range(len(in_texts)):
        step_i_completed = False
        while not step_i_completed:
            try:
                chunks.append(get_run_gpt_prompt(prompt, in_texts[i], model))
                step_i_completed = True
            except RateLimitError:
                print(f"exception RateLimitError at step {i}, waiting 5 sec and then continue")
                time.sleep(5)
                pass
        print(f"step {i}, completed")
        time.sleep(2)
    return chunks


def get_supported_summaries_from_aws():
    client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    return {"_".join(k['Key'].split("_")[0: -1]): k['Key']
            for k in client.list_objects(Bucket='ai-summarized-text')['Contents']
            if k['Key'].endswith(".csv")}


def get_df_from_aws(df_filename):
    client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    client.download_file('ai-summarized-text', df_filename, df_filename)
    df1 = pd.read_csv(df_filename, index_col='Unnamed: 0')
    return df1


def get_prompt_from_df(texts):
    context = "\n\n".join([t.replace("\n\n", "\n") for t in texts]).replace("  ", " ").strip()
    return f"""You are an AI assistant providing helpful advice.
You are given the following extracted parts of a long document and a question.
Provide an answer based on the context provided. Do NOT provide hyperlinks.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, or the context is empty, politely respond that you are tuned to only answer questions that are related to the context.
Answer in the same language of the question provided by the user.
Answer in Markdown.
Context:
{context}
"""


def query_origin_text(query, book_title):
    q_emb = st.session_state['model'].encode(query, convert_to_tensor=True)
    text_name = "text_summary" if st.session_state["text_for_query"] == "AI_extraction" else "text"
    emb_name = "emb_sum" if st.session_state["text_for_query"] == "AI_extraction" else "emb"
    st.session_state[book_title]["score_retrive_doc"] = st.session_state[book_title][emb_name].apply(lambda emb: util.pytorch_cos_sim(q_emb, emb)[0].numpy()[0])
    tmp = st.session_state[book_title][st.session_state[book_title].score_retrive_doc > 0.1].sort_values("score_retrive_doc", ascending=False)
    if tmp.shape[0] == 0:
        prompt = get_prompt_from_df([])
    else:
        for i in range(tmp.shape[0]):
            if num_tokens_from_string(get_prompt_from_df(tmp.iloc[0:i][text_name].values) + " " + query) > (MAX_TOKEN_MODEL * 0.75):
                break
        prompt = get_prompt_from_df(tmp.iloc[0:i-1][text_name].values)
    return get_run_gpt_prompt(prompt, query, model=gpt_model), tmp


def print_sources(sources, book_title):
    st.write("-"*75)
    st.write("Sources used reported below")
    text_name = "text_summary" if st.session_state["text_for_query"] == "AI_extraction" else "text"
    for src in sources[[text_name, "score_retrive_doc"]].iterrows():
        st.write("-"*75)
        block_description = "pages range in the pdf: **[{}]**".format(src[0])
        if 'epub' in st.session_state[book_title]:
            block_description = "chapter in the epub: **[{}]**".format(src[0])
        elif "article" in st.session_state[book_title]:
            block_description = "article chapter: **[{}]**".format(src[0])
        st.write(block_description + ". Retrive score: **{:0.2f}**".format(src[1].score_retrive_doc))
        st.write(src[1][text_name].replace("\n", "\n  "))
        st.write('\n  \n  ')


st.title('Ask a Question to the original book')
st.caption('Please remember to close this page in your browser when you have finished to question the book.')

if 'model' not in st.session_state:
    st.session_state['model'] = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

if "summarized_books" not in st.session_state:
    st.session_state["summarized_books"] = get_supported_summaries_from_aws()
book_title = st.selectbox("Choose the book", st.session_state["summarized_books"].keys())
st.session_state["text_for_query"] = st.selectbox("Choose source text", ["original", "AI_extraction"])

if book_title not in st.session_state:
    st.session_state[book_title] = get_df_from_aws(st.session_state["summarized_books"][book_title])
    st.session_state[book_title]["emb"] = st.session_state[book_title].text.apply(lambda x: st.session_state['model'].encode(x, convert_to_tensor=True))
    st.session_state[book_title]["emb_sum"] = st.session_state[book_title].text_summary.apply(lambda x: st.session_state['model'].encode(x, convert_to_tensor=True))

query = st.text_input("Enter a standalone question here (the bot is not aware of the previous questions asked, describe the context as much as possible). Use the same language of the book/summary.")
if query:
    answer, df_used = query_origin_text(query, book_title)
    st.markdown(answer)
    print_sources(df_used, book_title)
    del answer
    del df_used
    gc.collect()
