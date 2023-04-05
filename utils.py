import time

def format_time(time):
    '''
        Given a time in seconds, returns a string representing this
        time in the form hh:mm:ss.
    '''
    hours = time / (60**2)
    minutes = (hours - int(hours)) * 60
    secs = (minutes - int(minutes)) * 60

    return f"{str(int(hours)).zfill(2)}h:{str(int(minutes)).zfill(2)}m:{str(int(secs)).zfill(2)}s"

class Progress:
    def __init__(self, frequency: int, num_train_episodes: int) -> None:
        self.last_time_point = (0, 0) # (progress, time)
        self.start_time = time.time()
        self.checkpoint_start_time = self.start_time

        self.num_train_episodes = num_train_episodes
        self.frequency = frequency

    def print(self, episode):
        '''
            While training the model, this function can be used to
            print progress and estimated time.
        '''
        if episode % self.frequency != 0:
            return

        current_time = time.time() - self.start_time
        current_progress = episode/self.num_train_episodes
        if current_progress > 0:
            slope = (current_time - self.last_time_point[1]) / (current_progress - self.last_time_point[0])
            estimated_time = format_time(slope * (1 - current_progress))
            self.last_time_point = current_progress, current_time
        else:
            estimated_time = "None"

        print(f"Progress: {current_progress*100:.2f} % |"
            f"Tempo estimado restante: {estimated_time}\n")

    def set_checkpoint_start_time(self):
        self.checkpoint_start_time = time.time()

    def get_checkpoint_elapsed_time(self):
        return time.time() - self.checkpoint_start_time

if __name__ == "__main__":
    pass
    # print(format_time(60*60 * 3.423))
    # print_progress(10, 200, 1134)
    # print_progress(30, 200, 1134*3)
    # print_progress(60, 200, 1134*4)