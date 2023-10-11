from config import *
from langchain.llms import CTransformers
from langchain.evaluation import load_evaluator

class MistralEvaluator:
    
    def __init__(self):
        
        self.model = MODEL
        self.model_type = MODEL_TYPE
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE
        self.do_sample = DO_SAMPLE
        
        self.llm = CTransformers(
            model = self.model,
            model_type = self.model_type,
            max_new_tokens = self.max_new_tokens,
            temperature = self.temperature,
            do_sample = self.do_sample
        )
        
        self.criterias = [CORRECTNESS]
    
    def evaluate_criteria(self, criteria, input, prediction, reference):
        
        evaluator = load_evaluator(
                                    evaluator="labeled_criteria", 
                                    criteria=criteria,
                                    llm=self.llm
                                )
        
        eval_result = evaluator.evaluate_strings(
                                                    input=input,
                                                    prediction=prediction,
                                                    reference=reference
                                                )
        return eval_result["reasoning"]
    
    def eval_sample(self, input, prediction, reference):
        
        eval_output = {}
        
        for criteria in self.criterias:
            criteria_eval_sample = self.evaluate_criteria(
                                                            criteria=criteria,
                                                            input=input, 
                                                            prediction=prediction,
                                                            reference=reference
                                                        )
            eval_output[criteria] = criteria_eval_sample
        
        return eval_output