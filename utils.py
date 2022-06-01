def format_time(time):
    '''
        Given a time in seconds, returns a string representing this
        time in the form hh:mm:ss.
    '''

    hours = time / (60**2)
    minutes = (hours - int(hours)) * 60
    secs = (minutes - int(minutes)) * 60

    print(f"{str(int(hours)).zfill(2)}")
    return f"{str(int(hours)).zfill(2)}h:{str(int(minutes)).zfill(2)}m:{str(int(secs)).zfill(2)}s"

last_time_point = (0, 0) # (progress, time)
def print_progress(episode, num_train_episodes, current_time):
    '''
        While training the model, this function can be used to
        print progress and estimated time.
    '''

    global last_time_point

    current_progress = episode/num_train_episodes
    if current_progress > 0:
        slope = (current_time - last_time_point[1]) / (current_progress - last_time_point[0])
        estimated_time = format_time(slope * (1 - current_progress))
        last_time_point = current_progress, current_time
    else:
        estimated_time = "None"

    print(f"Progress: {current_progress*100:.2f} % |"
          f"Tempo estimado restante: {estimated_time}")


if __name__ == "__main__":
    # print(format_time(60*60 * 3.423))
    print_progress(10, 200, 1134)
    print_progress(30, 200, 1134*3)
    print_progress(60, 200, 1134*4)