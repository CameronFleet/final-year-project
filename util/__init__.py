def discretization_actions(Ft_discretization, Alpha_discretization, Fs_discretization):
    actions = []
    Ft_bounds = 0, 1.0
    alpha_bounds = -0.1, 0.1
    Fs_bounds = -1.0, 1.0

    for Ft in np.arange(Ft_bounds[0], Ft_bounds[1] + 0.001 , Ft_bounds[1]/Ft_discretization):

        if Ft != 0:
            for alpha in np.arange(alpha_bounds[0], alpha_bounds[1] + 0.001 , alpha_bounds[1]/Alpha_discretization):
                for Fs in np.arange(Fs_bounds[0], Fs_bounds[1] + 0.001, Fs_bounds[1]/Fs_discretization):
                    actions.append((Ft, alpha, Fs))

        if Ft == 0:
            for Fs in np.arange(Fs_bounds[0], Fs_bounds[1] + 0.001, Fs_bounds[1]/Fs_discretization):
                actions.append((Ft, 0, Fs))


    return actions