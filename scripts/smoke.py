from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "citylearn_challenge_2022_phase_1",
    central_agent=True,
    episode_time_steps=24,
)

observations, _ = env.reset()

print("num action spaces:", len(env.action_space))
print("action space[0]:", env.action_space[0])
print("observation container:", type(observations), len(observations))

steps = 0
while not env.terminated:
    actions = [env.action_space[0].sample()]
    result = env.step(actions)
    steps += 1

print(f"ok — {steps} steps, {len(result)} values returned by step()")
