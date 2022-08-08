import envs

if __name__ == '__main__':
    env = envs.WordleEnv2.create_default(render=True)
    env.reset()
    done = False
    env.print_render()
    while not done:
        i = input('guess > ')
        action = env._translate_word(i)
        _, reward, done, _ = env.step(action)
    if reward == 10:
        print('You won!')
    else:
        print(f'You lost: {env.word}')