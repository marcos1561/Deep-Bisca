# import tensorflow as tf
import matplotlib.pyplot as plt

from custom_env import np
import custom_env as env

class DebugModel:
    states_test = ["VAZIO- 5-e 1-e 8-p 2-c VAZIO- VAZIO- VAZIO- VAZIO- -1 -1",
                    "9-p 5-e 1-e 10-p 2-c VAZIO- VAZIO- VAZIO- VAZIO- -1 -1",
                    "VAZIO- 5-e VAZIO- 7-o 10-c 5-o 2-p VAZIO- VAZIO- 1 -1"]

    states_test_m = env.Bisca.human_to_machine(states_test) 
    states_test_m_rs = []
    for s in states_test_m:
        states_test_m_rs.append(env.Bisca.reshape_state(s))
    states_test_m_rs = np.array(states_test_m_rs)
    
    def __init__(self) -> None:
        self.predictions = []
        self.pred_ep = []

    def test_state(self, model, episode):
        ''' 
            Make the model give prediction for all the states in the static property 'states_m_rs'
            and store the result in the variable 'self.predictions'
        '''
        
        predicted = model.predict(DebugModel.states_test_m_rs, verbose=0)
        self.predictions.append(predicted)
        self.pred_ep.append(episode)


    def plot_predictions(self):
        '''
            Make a plot of the predictions stored in 'self.predictions'. 
            Each state has it's own figure and the plot consist in the values for each actions against
            the episode it was calculated.
        '''
        # Show the states in the console
        for id, s in enumerate(DebugModel.states_test_m):
            print(f"Estado {id}:")
            env.Bisca.print_state(s)
            print()

        # plot the predictions
        self.predictions = np.array(self.predictions)
        for s_id in range(len(DebugModel.states_test)):
            fig, ax = plt.subplots()
            ax.set_title(f"Estado {s_id}")

            s_p = self.predictions[:,s_id,:]
            for i in range(3):
                ax.plot(self.pred_ep, s_p[:, i], "-o", label=f"Predição {i}")
            ax.legend()
        plt.show()
            

if __name__ == "__main__":
    import tensorflow as tf

    debug_model = DebugModel()

    model = tf.keras.models.load_model("saved_model/" + "bianca_v6")
    
    debug_model.test_state(model, 1)
    debug_model.test_state(model, 10)

    debug_model.plot_predictions()
