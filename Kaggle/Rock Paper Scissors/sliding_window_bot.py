import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

my_plays = []
computer_plays = []
basic_pred_plays = []
random_opp_plays = []
bpb_score = 0
opp_score = 0
my_score = 0

def winner(you, other):
    if you == -1 or other == -1:
        return "X"
    
    if you == other:
        return "T "
    
    if you == 0 or you == 1: 
        if other == you + 1:
            return "L "
        else:
            return "W "
            
    else:
        if other == 0:
            return "L "
        else:
            return "W "

def random_agent_opposite (observation, configuration):
    import numpy as np
    shoot = np.random.randint(0,3)
    #Reverse the shoot.
    if shoot == 0:
        return 1
    elif shoot == 1:
        return 2
    else:
        return 0

def basic_pred(observation, configuration):
    global my_plays, computer_plays
    prevX = 5
    
    if observation.step > 20:
        prevX = 10
    # Now the fun begins
    trainData = []
    
    # Use as most the previous 30 entries for training
    trainEntryStart = 0
    if observation.step > 30:
        trainEntryStart = observation.step - 30
    for i in range(0,len(computer_plays) - prevX):
        curr_data = computer_plays[i:i+prevX+1]
        trainData.append(curr_data)
#             print(curr_data)

    df = pd.DataFrame(trainData)
    colnames = []
    for i in range(0,prevX):
        colnames.append("Prev " + str(prevX - i))
    colnames.append("target")
    df.columns = colnames

    modelOutcome = []
    df_train = df.iloc[0:int(len(df) * 0.7)]
    df_test = df.iloc[int(len(df) * 0.7):]
    for i in range(0,int(len(df) * 0.7)):
        modelOutcome.append("X")
    x_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    x_test = df_test.drop("target", axis=1)
    y_test = df_test["target"]

    random_forest = RandomForestClassifier(n_estimators=30)
    random_forest.fit(x_train, y_train)
    pred = random_forest.predict(x_test)
#         print(x_test)
#         print("Model prediction: ", pred)
    print("Basic pred model score: ", random_forest.score(x_test, y_test))

    #Create another dummy df for getting the actual prediction
    final_df = pd.DataFrame([computer_plays[-prevX:]])
    final_df.columns = colnames[0:-1]
    compMovePred = random_forest.predict(final_df)

#         print("Computer next move is predicted to be: ", compMovePred[0])
    if compMovePred[0] < 2:
        shoot = compMovePred[0] + 1
    else:
        shoot = 0
        
    print("Basic pred bot shooting: ", shoot)
    return int(shoot)
    
def showHistory(startEntry):
    global my_plays, computer_plays, random_opp_plays, basic_pred_plays, bpb_score, opp_score, my_score
    
    #Recalculate score from start Entry
    bpb_score = 0
    opp_score = 0
    my_score = 0
    
    me_result = "Me :"
    basic_pred_result = "BPB:"
    random_opp_result = "OPP:"
    
    for i in range(startEntry,len(computer_plays)):
        me_result += str(my_plays[i])
        basic_pred_result += str(basic_pred_plays[i])
        random_opp_result += str(random_opp_plays[i])
        
        curr_me = winner(my_plays[i], computer_plays[i]) 
        curr_bpb = winner(basic_pred_plays[i], computer_plays[i]) 
        curr_opp = winner(random_opp_plays[i], computer_plays[i]) 
        
        if curr_me[0] == "W":
            my_score += 1
        
        if curr_bpb[0] == "W":
            bpb_score += 1
    
        if curr_opp[0] == "W":
            opp_score += 1
    
#     print ("Plays:")
#     print("Me : ", my_plays)
#     print("Cpu: ", computer_plays)
#     print("BPB: ", basic_pred_plays)
#     print("OPP: ", random_opp_plays)
    
#     print ("\nCurrent results:")
#     print(me_result)
#     print(basic_pred_result)
#     print(random_opp_result)
    
#     print("\nCurrent scores:")
#     print("My score: ", my_score)
#     print("BPB score:", bpb_score)
#     print("Opp score:", opp_score)
    
def sliding_win(observation, configuration):
    global my_plays, computer_plays, random_opp_plays, basic_pred_plays
    
    prevX = 5 # Use the last 5 games to predict the next
    
    shoot = 0
    random_opp_shoot = 0
    basic_pred_shoot = 0
    
    compareStart = 0
    
    #Record computer play.
    if observation.step > 0:
        computer_plays.append(observation.lastOpponentAction)
        
    #What do we shoot?
    if observation.step < 10:
        shoot = np.random.randint(0,3)
    else:
        # Start using bot, and also random opp
        random_opp_shoot = random_agent_opposite(observation, configuration)
        basic_pred_shoot = basic_pred(observation, configuration)

        if observation.step < 20:
            shoot = basic_pred_shoot
        else:
            # See who has been the winnest in the last 20 plays.
            compareStart = 10
            if observation.step > 40:
                compareStart = observation.step - 20
                
            showHistory(compareStart)
            if bpb_score >= opp_score:
                shoot = basic_pred_shoot
            else:
                shoot = random_opp_shoot
            
        
    #Record history
    random_opp_plays.append(random_opp_shoot) 
    basic_pred_plays.append(basic_pred_shoot)
    my_plays.append(shoot)
    
#     print("Me : ", my_plays)
#     print("Cpu: ", computer_plays)
#     scores = ""
#     for i in range(0,len(computer_plays)):
#         scores += winner(my_plays[i], computer_plays[i]) 
#     print (scores)
#     print ("\n")
#     print("What is shoot? ", shoot)
    
    return int(shoot)