import numpy as np
from sentence_transformers import util
from collections import OrderedDict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


##### context 생성기 목록 ------------------------------------------------------------#


def multi_filter_and_cos(self, corpus, query, **kwargs):
    around_num = kwargs['around_num']

    query_encoding = self.encoder.encode(query,convert_to_tensor=True)
    corpus_encoding = self.encoder.encode(corpus,convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(query_encoding,corpus_encoding)[0].cpu().numpy()

    df = pd.DataFrame(corpus,columns=['paragraph'])

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform( [query]+corpus )
    freq_scores = np.array(tfidf.todense())[0,:]
    freqs = sorted(set(freq_scores))

    ind = freq_scores == -1000
    for i in range(3):
        try:
            ind = ind + (freq_scores == freqs[-1-i])
        except:
            pass
    vocab = vectorizer.get_feature_names_out()
    unique_vocab = vocab[ind]
    m = len(unique_vocab)

    l = len(self.encoder.tokenizer.tokenize(query))
    N = len(corpus)
    vocab_ratio = np.zeros(N)
    token_ratio = np.zeros(N)
    for i in range(N):
        par = corpus[i]
        L = len(self.encoder.tokenizer.tokenize(par))
        ct1 = 0
        for vocab in unique_vocab:
            if vocab in par:
                ct1 += 1
        vocab_ratio[i] = ct1/m
        token_ratio[i] = L/l
        
    df = df.assign(token_ratio = token_ratio,
                   vocab_ratio = vocab_ratio,
                   cos_sim = cos_sim)
    
    df['word_include_flag'] = df['paragraph'].str.\
        contains('표 |차트 |그림 |출처:', regex=True).astype(int)

    filter_df = df.loc[(df['vocab_ratio'] > 0) \
        & (df['token_ratio'] >= 0.5) \
        & (df['word_include_flag'] != 1), :]

    candidates = filter_df.sort_values(by=['vocab_ratio','cos_sim'], 
                                       axis=0, ascending=False).head(3)
    
    filter_df_2 = df.loc[(df['token_ratio'] >= 0.5) \
        & (df['word_include_flag'] != 1), :]

    rank_inds = np.array(candidates.index)
    filter_df_2_inds = np.array(filter_df_2.index)

    contexts = OrderedDict({})
    context_ids = OrderedDict({})
    center_ids = OrderedDict({})
    rank_ct = 0
    for rank_ind in rank_inds:
        rank_ct += 1
        context = []
        ids = []
        ind = filter_df_2_inds[filter_df_2_inds == rank_ind][0]
        iloc_ct = -1
        for df_2_ind in filter_df_2_inds:
            iloc_ct += 1
            if df_2_ind == ind:
                break
        for j in range(-around_num,around_num+1):
            try:
                context.append(filter_df_2.iloc[iloc_ct+j]['paragraph'])
                ids.append(filter_df_2_inds[iloc_ct+j])
            except:
                pass
        rank_list_idx = ids.index(rank_ind)
        contexts.update({f'{rank_ct}': context})
        context_ids.update({f'{rank_ct}': ids})
        center_ids.update({f'{rank_ct}': rank_list_idx})
        
    return {
        'contexts': contexts,
        'context_ids': context_ids,
        'center_ids': center_ids
    }
    

def collect_high_cos_sim(pipeline, corpus, query, **kwargs):
    issues = []
    collect_num = kwargs['collect_num']
    ignore_short_sentences = kwargs['ignore_short_sentences']
    minimal_token_length = kwargs['minimal_token_length']
    
    query_encoding = pipeline.encoder.encode(query, convert_to_tensor=True)

    def plain_corpus_encoding():
        parsed_plain = np.array([])
        for value in corpus:
            if ignore_short_sentences is True:
                sentence_tokens = pipeline.encoder.tokenizer.tokenize(value)
                if len(sentence_tokens) >= minimal_token_length:
                    parsed_plain = np.append(parsed_plain, value)
            else:
                parsed_plain = np.append(parsed_plain, value)
        corpus_plain_encoding = pipeline.encoder.encode(parsed_plain, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(
            query_encoding, corpus_plain_encoding).cpu().numpy()[0]
        return parsed_plain, scores

    parsed_plain, scores = plain_corpus_encoding()

    max_ids = np.flip(np.argsort(scores))[:collect_num]
    query_context = []
    for id in max_ids:
        query_context.append(parsed_plain[id])

    pipeline.solving_progress.update({'query_context': query_context}) # optional

    return query_context


def sim_high_n_and_following_2(self, corpus, query, **kwargs):
    
    n = kwargs['n']
    ignore_short_sentences = kwargs['ignore_short_sentences']
    minimal_token_length = kwargs['minimal_token_length']
    
    query_encoding = self.encoder.encode(query, convert_to_tensor=True)
    
    def plain_corpus_encoding():
        parsed_plain = np.array([])
        for _, value in corpus.items():
            if ignore_short_sentences is True:
                sentence_tokens = self.encoder.tokenizer.tokenize(value)
                if len(sentence_tokens) >= minimal_token_length:
                    parsed_plain = np.append(parsed_plain, value)
            else:
                parsed_plain = np.append(parsed_plain, value)
        corpus_plain_encoding = self.encoder.encode(parsed_plain, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(
            query_encoding, corpus_plain_encoding).cpu().numpy()[0]
        return parsed_plain, scores
    
    parsed_plain, scores = plain_corpus_encoding()
    
    max_ids = np.flip(np.argsort(scores))[:n]
    query_contexts = {}
    rank = -1
    for id in max_ids:
        rank += 1
        query_contexts.update({f'{rank}': parsed_plain[id:id+3]})
        
    return query_contexts


def chunk_n_by_sim(pipeline, corpus, query, **kwargs):
    n = kwargs['n']
    around_num = kwargs['around_num']
    ignore_short_sentences = kwargs['ignore_short_sentences']
    minimal_token_length = kwargs['minimal_token_length']

    query_encoding = pipeline.encoder.encode(query, convert_to_tensor=True)

    def plain_corpus_encoding():
        parsed_plain = np.array([])
        for value in corpus:
            if ignore_short_sentences is True:
                sentence_tokens = pipeline.encoder.tokenizer.tokenize(value)
                if len(sentence_tokens) >= minimal_token_length:
                    parsed_plain = np.append(parsed_plain, value)
            else:
                parsed_plain = np.append(parsed_plain, value)
        corpus_plain_encoding = pipeline.encoder.encode(parsed_plain, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(
            query_encoding, corpus_plain_encoding).cpu().numpy()[0]
        return parsed_plain, scores

    parsed_plain, scores = plain_corpus_encoding()

    max_ids = np.flip(np.argsort(scores))[:n]

    query_contexts = {}
    context_ct = -1
    parsed_plain_all = list(corpus)
    print('total corpus length:', len(parsed_plain_all))

    for id in max_ids:
        context_ct += 1
        center_sentence = parsed_plain[id]
        for j, corpus_sentence in enumerate(parsed_plain_all):
            if center_sentence == corpus_sentence:
                context = []
                print('j:', j)
                for k in range(-around_num, around_num + 1):
                    try:
                        context.append(parsed_plain_all[j + k])
                    except:
                        pass

                query_contexts.update({f'{context_ct}': context})

    pipeline.solving_progress.update({'query_context': query_contexts})  # optional

    return query_contexts
    
    
##### task1_solver 목록  ------------------------------------------------------------#

def direct_fill(self, corpus, context, query, answer_form, **kwargs):
    contexts = context['contexts']
    center_ids = context['center_ids']
    
    for i in range(0,3):
        try:
            answer = contexts[f'{i+1}'][center_ids[f'{i+1}']]
            answer_form[i]['paragraph'] = answer
        except:
            pass
        
    return answer_form
  

def simple_similarity(self, context, query, answer_form, **kwargs):
    
    for i in range(0,3):
        answer_form[i]['paragraph'] = context[i]
    
    return answer_form


def qa_model_or_similarity(self, context, query, answer_form, **kwargs):
    issues = []
    contexts = context

    def plain_sts_match():
        query_encoding = self.encoder.encode(query, convert_to_tensor=True)
        context_encoding = self.encoder.encode(context, convert_to_tensor=True)
        context_scores = util.pytorch_cos_sim(query_encoding,
                                              context_encoding).cpu().numpy()[0]
        return context[np.argmax(context_scores)]
    
    for i in range(3):
        context = contexts[f'{i}']

        #print(context)
        inputs = {"context": ". ".join(context), "question": query}
        QA_model_result = self.qa_model.predict_answer([inputs])
        
        # QA_model_result = self.qa_model.predict_answer(context=". ".join(context),
        #                                                question=query)
        QA_model_answer = QA_model_result['answer_text']
        
        tokenizer = self.encoder.tokenizer
        
        if QA_model_answer is not None:
            answer_text_token_ids = tokenizer(QA_model_answer)['input_ids']
            inferred_answer = None

            for item in context:  # QA_model의 토큰과 context의 토큰 비교
                item_token_ids = tokenizer(item)['input_ids']
                if list(set(answer_text_token_ids).difference(item_token_ids)) == []:
                    inferred_answer = item
                    break

            if inferred_answer is None:  # paragraph 찾기에 실패한 경우, STS로 대체
                print(
                    "QA_model: Failed to match paragraph. Retrying with STS-method ...")
                issues.append('paragraph_match_fail')
                inferred_answer = plain_sts_match()

            self.solving_progress.update({'QA_result': QA_model_answer})  # optional
            self.solving_progress.update({'inferred_answer': inferred_answer})
        else:
            print("QA_model: Failed to match paragraph. Retrying with STS-method ...")
            issues.append('paragraph_match_fail')
            inferred_answer = plain_sts_match()
        
        answer_form[i]['paragraph'] = inferred_answer

    return answer_form