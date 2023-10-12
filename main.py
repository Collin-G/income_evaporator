from model import Model
from stacker_model import StackerModel
company = "TSLA"

long_predict = 60
med_predict = 30
short_predict = 7

model1 = Model(company, long_predict)
model2 = Model(company, med_predict)
model3 = Model(company, short_predict)


stacked = StackerModel(model1, model2,model3)
stacked.plot_results()
