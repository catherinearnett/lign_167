import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from hanziconv import HanziConv

#stims
exp1 = pd.read_csv("vv_pilot_stims.csv") #vv pilot may 2019
#exp2 = pd.read_csv("bounded_stims_final.csv") #vv boundedness follow up (jan 2020?)
exp3 = pd.read_csv("exp_3_stims.csv") # more instances, more duration, more; 4 answers

data = [
['IDEA-CCNL/Wenzhong-GPT2-110M', '110M', 'gpt2'],
['IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese', '3.5B', 'gpt2'],
['bigscience/bloom-560m', '560M', 'bloom'],
['bigscience/bloom-1b1', '1.1B', 'bloom'],
['bigscience/bloom-3b', '3B', 'bloom'],
#['bigscience/bloom-7b1', '7.1B', 'bloom', "bigscience/bloom-7b1"],
['facebook/xglm-564M', '564M', 'xglm'],
['facebook/xglm-1.7B', '1.7B', 'xglm'],
['facebook/xglm-2.9B', '2.9B', 'xglm'],
['facebook/xglm-4.5B', '4.5B', 'xglm']]
#['facebook/xglm-7.5B', '7.5B', 'xglm', "facebook/xglm-7.5B"]]

models = pd.DataFrame(data, columns = ['model_name', 'size', 'type'])

def tidy_gen_text(prompt, text):
    test_cleaned = str(text[0])
    test_cleaned = test_cleaned.replace(" ", "")
    test_cleaned = test_cleaned.replace(prompt, "")
    test_cleaned = HanziConv.toSimplified(test_cleaned)
    return test_cleaned


#together
exp1_results = pd.DataFrame(columns = ['model', 'option_a', 'option_b', 'response'])
exp1_results.to_csv("exp_1_with_all_results.csv")
# exp2_results = pd.DataFrame(columns = ['model', 'option_a', 'option_b', 'response'])
# exp2_results.to_csv("exp_2_with_all_results.csv")
exp3_results = pd.DataFrame(columns = ['model', 'option_a', 'option_b', 'duration_response', 'instances_response'])
exp3_results.to_csv("exp_3_with_all_results.csv")
for m in range(len(models)):
    model = models.loc[m]['model_name']
    print(model)
    size = models.loc[m]['size']
    type_m = models.loc[m]['type']
    col_name = str(type_m+"_"+size+"_response")
    generator = pipeline(task="text-generation", model=model, max_new_tokens = 50, num_return_sequences=1, device=0)
    #experiment 1
    for i in range(5): #len(exp1)
        option_A = exp1.loc[i]['V']
        option_B = exp1.loc[i]['VV']
        full_prompt = "下列哪个活动所需时间更长? \nA: 跑马拉松 \nB: 刷牙 \n回答: A: 跑马拉松 \n 下列哪个活动所需时间更长? \nA: 买菜 \nB: 眨眼 \n回答: A: 买菜 \n 下列哪个活动所需时间更长? \nA: 关电灯 \nB: 看电影 \n回答: B: 看电影 \n下列哪个活动所需时间更长? \nA:" +option_A + "\nB:" + option_B +"\n回答:"
        response = generator(full_prompt)
        response = tidy_gen_text(full_prompt,response)
        print(response)
        with open('testing.txt', 'a') as f:
            f.write(response + '\n')
        if "A:" in response:
            response_type = "A"
        elif "B:" in response:
            response_type = "B"
        data_point = pd.DataFrame({'model': [col_name], 'option_a': [option_A], 'option_b': [option_B],'response': [response_type]})
        data_point.to_csv('exp_1_with_all_results.csv', mode='a', index=False, header=False)
    #experiment 2
    # for i in range(len(exp2)): #
    #     stimulus = exp2.loc[i]['form']
    #     option_A = exp2.loc[i]['option_a']
    #     option_B = exp2.loc[i]['option_b']
    #     if type(option_A) == str and type(option_B) == str:
    #         full_prompt = "玛丽在商店买了面包 \n给定上面的句子, 下列哪个活动所需时间更长? \nA: 玛丽买了面包 \nB: 玛丽没有买面包 \n回答: A: 玛丽买了面包 \n" +stimulus+"\nA:" +option_A + "\nB:" + option_B +"\n回答:"
    #         response = generator(full_prompt)
    #         response = tidy_gen_text(full_prompt,response)
    #         if "A:" in response:
    #             response_type = "A"
    #         elif "B:" in response:
    #             response_type = "B"
    #         # exp2_results.loc[len(exp1_results)] = [col_name, option_A, option_B, response_type]
    #         # data = [col_name, option_A, option_B, response_type]
    #         data_point = pd.DataFrame({'model': [col_name], 'option_a': [option_A], 'option_b': [option_B],'response': [response_type]})
    #         # data_point = pd.DataFrame(data, columns = ['model', 'option_a', 'option_b', 'response'])
    #         data_point.to_csv('exp_2_with_all_results.csv', mode='a', index=False, header=False)
    #experiment 3
    for i in range(len(exp3)): #
        option_A = exp3.loc[i]['a_option']
        option_B = exp3.loc[i]['b_option']
        duration_prompt = exp3.loc[i]['duration_prompt']
        instances_prompt = exp3.loc[i]['instances_prompt']
        full_duration_prompt = "A或B的时间多? \nA: 跑马拉松 \nB: 刷牙 \n回答: A: 跑马拉松  \n" + duration_prompt+ "? \n" +option_A + "\n" + option_B +"\n回答:"
        response = generator(full_duration_prompt)
        response = tidy_gen_text(full_duration_prompt,response)
        if "A:" in response:
            response_type = "A"
        elif "B:" in response:
            response_type = "B"
        full_instances_prompt = "A或B的时间多? \nA: 跑马拉松 \nB: 刷牙 \n回答: A: 跑马拉松  \n" + instances_prompt+ "? \n" +option_A + "\n" + option_B +"\n回答:"
        response2 = generator(full_instances_prompt)
        response2 = tidy_gen_text(full_instances_prompt,response2)
        if "A:" in response2:
            response_type2 = "A"
        elif "B:" in response2:
            response_type2 = "B"
    data_point = pd.DataFrame({'model': [col_name], 'option_a': [option_A], 'option_b': [option_B], 'duration_response': [response_type], 'instances_response': [response_type2]})
    data_point.to_csv('exp_3_with_all_results.csv', mode='a', index=False, header=False)

exp1_results.to_csv('exp_1_with_all_results.csv', mode='a', index=False, header=False)
# exp2_results.to_csv("exp_2_with_all_results.csv", mode='a', index=False, header=False)
exp3_results.to_csv("exp_3_with_all_results.csv", mode='a', index=False, header=False)
