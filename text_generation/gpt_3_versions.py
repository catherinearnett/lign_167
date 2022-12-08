import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import openai

from hanziconv import HanziConv

open_ai_api = "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"
openai.organization = "org-s8IwVcAXrel9VPNuOdcXP4R2"
openai.api_key = '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'


#stims
exp1 = pd.read_csv("vv_pilot_stims.csv") #vv pilot may 2019
exp2 = pd.read_csv("bounded_stims_final.csv") #vv boundedness follow up (jan 2020?)
exp3 = pd.read_csv("exp_3_stims.csv") # more instances, more duration, more; 4 answers

#versions = ['text-davinci-003', 'text-davinci-002', 'text-davinci-001', 'davinci']

def openAIQuery_03(query):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=query,
      temperature=0.8,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']
        else:
            answer = 'Opps sorry, you beat the AI this time'
    else:
        answer = 'Opps sorry, you beat the AI this time'

    return answer


def openAIQuery_02(query):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=query,
      temperature=0.8,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']
        else:
            answer = 'Opps sorry, you beat the AI this time'
    else:
        answer = 'Opps sorry, you beat the AI this time'

    return answer

def openAIQuery_01(query):
    response = openai.Completion.create(
      engine="text-davinci-001",
      prompt=query,
      temperature=0.8,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']
        else:
            answer = 'Opps sorry, you beat the AI this time'
    else:
        answer = 'Opps sorry, you beat the AI this time'

    return answer

def openAIQuery_00(query):
    response = openai.Completion.create(
      engine="davinci",
      prompt=query,
      temperature=0.8,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0)

    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']
        else:
            answer = 'Opps sorry, you beat the AI this time'
    else:
        answer = 'Opps sorry, you beat the AI this time'

    return answer


exp_1_gpt3_results = pd.DataFrame(columns=['model', 'option_A', 'response_type', 'response'])
exp_1_gpt3_results.to_csv('exp_1_gpt3_results.csv')
for m in [openAIQuery_03, openAIQuery_02, openAIQuery_01, openAIQuery_00]:
    for i in range(len(exp1)):
        option_A = exp1.loc[i]['V']
        option_B = exp1.loc[i]['VV']
        full_prompt = "下列哪个活动所需时间更长? \nA: 跑马拉松 \nB: 刷牙 \nAnswer: A: 跑马拉松  \n下列哪个活动所需时间更长? \nA:" +option_A + "\nB:" + option_B +"\nAnswer:"
        response = m(full_prompt)
        if "A:" in response:
            response_type = "A"
        elif "B:" in response:
            response_type = "B"
        # response_types.append(response_type)
        data_point = pd.DataFrame({'model': [str(m)], 'option_a': [option_A], 'option_b': [option_B],'response_type': [response_type], 'response': [str(response)]})
        data_point.to_csv('exp_1_gpt3_results.csv', mode='a', index=False, header=False)

#experiment2
exp_2_gpt3_results = pd.DataFrame(columns=['model', 'option_A', 'response_type', 'response'])
exp_2_gpt3_results.to_csv('exp_2_gpt3_results.csv')
for m in [openAIQuery_03, openAIQuery_02, openAIQuery_01, openAIQuery_00]:
    for i in range(len(exp2)):
        stimulus = exp2.loc[i]['form']
        option_A = exp2.loc[i]['option_a']
        option_B = exp2.loc[i]['option_b']
        if type(option_A) == str and type(option_B) == str:
            full_prompt = "玛丽在商店买了面包 \n给定上面的句子, 下列哪个活动所需时间更长? \nA: 玛丽买了面包 \nB: 玛丽没有买面包 \n回答: A: 玛丽买了面包 \n" +stimulus+"\nA:" +option_A + "\nB:" + option_B +"\n回答:"
            response = m(full_prompt)
            if "A:" in response:
                response_type = "A"
            elif "B:" in response:
                response_type = "B"
        # response_types.append(response_type)
        data_point = pd.DataFrame({'model': [str(m)], 'option_a': [option_A], 'option_b': [option_B],'response_type': [response_type], 'response': [str(response)]})
        data_point.to_csv('exp_2_gpt3_results.csv', mode='a', index=False, header=False)

#experiment 3
exp_3_gpt3_results = pd.DataFrame(columns=['model', 'option_A', 'duration_response_type', 'duration_response', 'instances_response_type', 'instances_response'])
exp_3_gpt3_results.to_csv('exp_3_gpt3_results.csv')
for m in [openAIQuery_03, openAIQuery_02, openAIQuery_01, openAIQuery_00]:
    for i in range(len(exp3)):
        option_A = exp3.loc[i]['a_option']
        option_B = exp3.loc[i]['b_option']
        duration_prompt = exp3.loc[i]['duration_prompt']
        instances_prompt = exp3.loc[i]['instances_prompt']
        full_duration_prompt = "A或B的时间多? \nA: 跑马拉松 \nB: 刷牙 \n回答: A: 跑马拉松  \n" + duration_prompt+ "? \n" +option_A + "\n" + option_B +"\n回答:"
        response = m(full_duration_prompt)
        if "A:" in response:
            response_type = "A"
        elif "B:" in response:
            response_type = "B"
        full_instances_prompt = "A或B的时间多? \nA: 跑马拉松 \nB: 刷牙 \n回答: A: 跑马拉松  \n" + instances_prompt+ "? \n" +option_A + "\n" + option_B +"\n回答:"
        response2 = m(full_instances_prompt)
        if "A:" in response2:
            response_type2 = "A"
        elif "B:" in response2:
            response_type2 = "B"
        data_point = pd.DataFrame({'model': [str(m)], 'option_a': [option_A], 'option_b': [option_B],'duration_response_type': [response_type], 'duration_response': [str(response)],'instances_response_type': [response_type2], 'instances_response': [str(response2)]})
        data_point.to_csv('exp_3_gpt3_results.csv', mode='a', index=False, header=False)
