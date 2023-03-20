import envs

if __name__ == "__main__":
    env = envs.WordleEnv4.create_default(render=True, max_sub_attempts=10, prefill=3)
    env.reset()
    done = False
    env.print_render()
    while not done:
        i = input("guess > ")
        i = i.ljust(5 - env.prefill, "x")[: 5 - env.prefill]
        print(i)
        if len(i) > 1:
            for a in i[:-1]:
                action = env.index_map.get(a)
                _, reward, done, _ = env.step(action, skip_render=True)
        action = env.index_map.get(i[-1])
        _, reward, done, _ = env.step(action)
    if reward == 10:
        print("You won!")
    else:
        print(f"You lost: {env.word}")
