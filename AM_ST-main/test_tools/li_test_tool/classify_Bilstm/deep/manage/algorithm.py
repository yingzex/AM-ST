﻿import heapq
import math
from multiprocessing.dummy import Pool as ThreadPool
import string
from test_tools.li_test_tool.classify_Bilstm.deep.dataloader.util import *



def greed(sentence, cr, deploy_model):
    """
    Each step get the most probable word until <END>.
    """
    minLen = 5
    maxLen = 20
    (x, x_mask) = cr.transform_input_data(sentence)
    print("frequency_ratio x: ", sentence)
    currentLength = 0
    
    error = 0.0
    
    while True: 
        pred_w, pred_w_p = deploy_model(x, x_mask)
        pred_w_list = pred_w_p.flatten().tolist()
        
        sorted_index = heapq.nlargest(1, enumerate(pred_w_list), key=lambda s: s[1])
        p_word, p_word_prob = zip(*(sorted_index[:10]))
        # print 'currentLength %d:' % currentLength
        l1 = cr.transformInputText(p_word)
        l2 = p_word_prob
        # for ll1, ll2 in zip(l1, l2):
        #    print '%s\t%.8f' % (ll1, ll2)
        error -= numpy.log(l2)
        
        if currentLength < minLen:
            for c in p_word:
                choice = c
                if choice > 2:
                    break
        else:
            for c in p_word:
                choice = c
                if choice >= 2:
                    break
                
        x = numpy.concatenate([x, [[choice]]], axis=0)
        
        x_mask = numpy.concatenate([x_mask, [[1]]], axis=0)
        currentLength += 1
        if choice == 2 or currentLength > maxLen:
            break
#             print "pred: ", x
    return [x.flatten().tolist()[0]], error


def beam_search(sentence, cr, deploy_model, minLen=1, maxLen = 50, search_scope = 200,
                beam_size = 200, output_size = 20):
    """
    Each step get the most x probable word, Then explain the x word each step until <END>.
    """
    pool = ThreadPool(12)
    stop_tag = cr.get_word_dictionary()['<END>']
    available_flag = 10000000
   
    class SentenceScorePair(object):
        def __init__(self, priority, sentence):
            self.priority = priority
            self.sentence = sentence
        def __cmp__(self, other):
            return cmp(self.priority, \
                       other.priority)
    
    def step(last_sentence, score, pred_words_prob):
        pred_words_list = pred_words_prob.flatten().tolist()
        sorted_index = heapq.nlargest(search_scope, enumerate(pred_words_list), key=lambda s: s[1])
        results = list()
        cands = list()
        for pred_word, pred_word_prob in sorted_index:
            current_score = score - math.log(pred_word_prob)
            if available_flag < current_score:
                continue
            new_sentence = last_sentence + [pred_word]
            
            if pred_word == stop_tag:
                if len(new_sentence) - base_length < minLen:
                    continue
                cands.append(SentenceScorePair(score - math.log(pred_word_prob), new_sentence))
            else:
                results.append(SentenceScorePair(score - math.log(pred_word_prob), new_sentence))
        return results, cands
    
    # get question and question_mask
    (question, question_mask) = cr.transform_input_data(sentence)
    base_length = question.shape[0]
    (tanswer, tanswer_mask) = (question[-1:,:], question_mask[-1:,:])
    pred_word, pred_words_prob = deploy_model(question[:-1,:], question_mask[:-1,:], tanswer, tanswer_mask)
    sentence = load_sentence('', cr.word2index, special_flag=cr.special_flag)
    sQueue, candidates = step(sentence, 0, pred_words_prob)
    # search_scope = 1
    # iterative from current sentence get new sentence by step() 
    for iter in xrange(maxLen):
        current_len = len(sQueue)
        if current_len == 0:
            break
        buffer_Queue = list()
        candidate_list = [q.sentence for q in sQueue]
        current_len = len(candidate_list)
        
        tanswer, tanswer_mask = get_mask_data(candidate_list)
        ext_question = numpy.concatenate([question]*tanswer.shape[1], axis=1)
        ext_question_mask = numpy.concatenate([question_mask]*tanswer.shape[1], axis=1)
        pred_word, pred_words_prob = deploy_model(ext_question[:-1,:], ext_question_mask[:-1,],
                                                  tanswer, tanswer_mask)
        # multi thread
        sentences = pool.map(lambda i: step(sQueue[i].sentence, sQueue[i].priority,
                                            pred_words_prob[i]),
                             range(current_len))
        for sentence, priority in sentences:
            buffer_Queue.extend(sentence)
            candidates.extend(priority)
            
        if len(candidates) >= output_size:
            candidates = heapq.nsmallest(output_size, candidates)
            available_flag = candidates[-1].priority
        sQueue = buffer_Queue
        sQueue = heapq.nsmallest(beam_size, sQueue)
    pool.close()
    
    candidates = heapq.nsmallest(output_size, candidates, key=lambda x: 1.0 * x.priority)# / len(x.sentence))
    return [cand.sentence for cand in candidates], [cand.priority for cand in candidates]


def beam_search_t(sentence, cr, deploy_model,answer_dict, minLen=2, maxLen = 50, search_scope = 200,
                beam_size = 200, output_size = 10):
    """
    Each step get the most x probable word, Then explain the x word each step until <END>.
    """
    pool = ThreadPool(1)
    stop_tag = cr.get_word_dictionary()['<END>']
    available_flag = 10000000
    #beam_small_value=100000000

    class SentenceScorePair(object):
        def __init__(self, priority, sentence):
            self.priority = priority
            self.sentence = sentence
        def __cmp__(self, other):
            return cmp(self.priority, \
                       other.priority)

    def step(last_sentence, score, pred_words_prob_array_old,pred_word_array_old,beam_small_value,beam_full_flag):
        pred_words_prob_array=pred_words_prob_array_old.flatten()
        pred_word_array=pred_word_array_old.flatten()
        #pred_words_list = pred_words_prob.flatten().tolist()
        #sorted_index = heapq.nlargest(search_scope, enumerate(pred_words_list), key=lambda s: s[1])
        results = list()
        cands = list()
        #for pred_word, pred_word_prob in sorted_index:
        for i in range(len(pred_word_array)):
            pred_word_prob=pred_words_prob_array[i]+0.00001
            pred_word=pred_word_array[i]
            current_score = score - math.log(pred_word_prob)
            if available_flag < current_score/(len(last_sentence)+1):
                pass
                continue
            new_sentence = last_sentence + [pred_word]
            #print 'model:',str(new_sentence)
            #print 'reference:',answer_dict.keys()[3]
            if answer_dict.get(str(new_sentence))==None:
                pass
                continue
            if pred_word == stop_tag:
                if len(new_sentence) < minLen:
                    continue
                #cands.append(SentenceScorePair(score - math.log(pred_word_prob), new_sentence))
                cands.append(SentenceScorePair((score - math.log(pred_word_prob))/len(new_sentence), new_sentence))
            else:
                
                if(beam_full_flag and beam_small_value<score - math.log(pred_word_prob)):
                    continue
                
                beam_small_value=max(beam_small_value,score - math.log(pred_word_prob))
                results.append(SentenceScorePair(score - math.log(pred_word_prob), new_sentence))
        return results, cands,beam_small_value

    # get question and question_mask
    sentence_topic=sentence.strip().split('\t')
    topic_label=[[string.atoi(sentence_topic[1])]]
    sentence=sentence_topic[0]
    (question, question_mask) = cr.transform_input_data(sentence)
    base_length = question.shape[0]
    (tanswer, tanswer_mask) = (question[-1:,:], question_mask[-1:,:])
    pred_word, pred_words_prob = deploy_model(question[:-1,:], question_mask[:-1,:], tanswer, tanswer_mask,topic_label)
    sentence = load_sentence('', cr.word2index, special_flag=cr.special_flag)
    sentence=[sentence[1]]
    sQueue, candidates,beam_value = step(sentence, 0, pred_words_prob,pred_word,0,0)
    # search_scope = 1
    # iterative from current sentence get new sentence by step()
    for iter in xrange(maxLen):
        current_len = len(sQueue)
        if current_len == 0:
            break
        buffer_Queue = list()
        candidate_list = [q.sentence for q in sQueue]
        current_len = len(candidate_list)

        tanswer, tanswer_mask = get_mask_data(candidate_list)
        ext_question = numpy.concatenate([question]*tanswer.shape[1], axis=1)
        ext_question_mask = numpy.concatenate([question_mask]*tanswer.shape[1], axis=1)
        ext_topic = numpy.concatenate([topic_label]*tanswer.shape[1], axis=1).astype('float32')
        pred_word, pred_words_prob = deploy_model(ext_question[:-1,:], ext_question_mask[:-1,],
                                                  tanswer, tanswer_mask,ext_topic)
        #print pred_words_prob
        # multi thread
        '''
        sentences = pool.map(lambda i: step(sQueue[i].sentence, sQueue[i].priority,
                                            pred_words_prob[i],pred_word[i]),
                             range(current_len))
        '''
        sentences=[]
        beam_small_value=0
        for i in range(current_len):
            results, cands,beam_small_value=step(sQueue[i].sentence, sQueue[i].priority, pred_words_prob[i],pred_word[i],beam_small_value,len(sentences)>beam_size)
            sentences.append((results,cands))
        for sentence, priority in sentences:
            buffer_Queue.extend(sentence)
            candidates.extend(priority)

        if len(candidates) >= output_size:
            candidates = heapq.nsmallest(output_size, candidates)
            available_flag = candidates[-1].priority
        sQueue = buffer_Queue
        sQueue = heapq.nsmallest(beam_size, sQueue)
    pool.close()

    candidates = heapq.nsmallest(output_size, candidates, key=lambda x: 1.0 * x.priority)# / len(x.sentence))
    return [cand.sentence for cand in candidates], [cand.priority for cand in candidates]
