from park.param import config


def generate_job(np_random, job_distribution):
    if job_distribution == 'Pareto':
        size = int((np_random.pareto(
                config.job_size_pareto_shape) + 1) * \
                config.job_size_pareto_scale)
    elif job_distribution == 'Uniform': # Ref: https://github.com/kaustubhsridhar/input_driven_rl_example/blob/master/load_balance/job_generator.py
        size = int(np_random.uniform(
                config.job_size_min, config.job_size_max))
    else:
        raise ValueError('Unknown job distribution')

    t = int(np_random.exponential(config.job_interval))

    return t, size

def generate_jobs(num_stream_jobs, np_random, job_distribution):

    # time and job size
    all_t = []
    all_size = []

    # generate streaming sequence
    t = 0
    for _ in range(num_stream_jobs):
        dt, size = generate_job(np_random, job_distribution)
        t += dt
        all_t.append(t)
        all_size.append(size)

    return all_t, all_size
