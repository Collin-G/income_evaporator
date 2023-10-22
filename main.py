from model import Model
from stacker_model import StackerModel
company = "TSLA"

long_predict = 60
med_predict = 30
short_predict = 7

model1 = Model(company, long_predict,20)
model2 = Model(company, med_predict,20)
model3 = Model(company, short_predict,20)


stacked = StackerModel(model1, model2,model3)
# stacked.get_accuracy()
stacked.plot_results()

