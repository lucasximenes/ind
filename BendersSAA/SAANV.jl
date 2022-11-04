using JuMP, Plots, HiGHS, Distributions, Statistics, PlotlyJS
plotlyjs()

# Define the second stage problem and f(x) to make it easier to evaluate the cost of buying x newspapers. 
function Second_Stage(x,d)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variables(m, 
    begin
        y ≥ 0
        w ≥ 0
    end)

    @constraints(m,
    begin
        y ≤ d
        y + w ≤ x
    end)

    @objective(m, Min, -25 * y - 5 * w)
    optimize!(m)
    return objective_value(m)
end

f(x, d) = 10*x + Second_Stage(x,d)

#SAA for the newsvendor problem

# M is the number of replicas of the demand distribution
M = 10
# possible amounts of samples to be drawn from the distribution for the lower bound
N_amounts = collect(50:50:1000)
cont_unif = Uniform(50, 150)



function Q(d)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    n = length(d)
    @variables(m,
    begin
        0 ≤ x ≤ 150
        y[1:n] ≥ 0
        w[1:n] ≥ 0
    end)
    
    @constraints(m,
    begin
        ct1[s=1:n], y[s] ≤ d[s]
        ct2[s=1:n], y[s] + w[s] ≤ x
    end)
    
    @objective(m, Min, 10 * x + (1/n)*sum((-25*y[s] -5*w[s]) for s = 1:n))
    
    optimize!(m)
    
    return objective_value(m), value(x)
end

LB_V = zeros(length(N_amounts))
Mean_V = zeros(length(N_amounts))
std_V = zeros(length(N_amounts), 2)


for (index, N) in enumerate(N_amounts)
    demand_matrix = rand(cont_unif, M, N)
    lower_bound = 0
    x_mean = 0
    lower_bound_samples = zeros(M)
    x_samples = zeros(M)
    for i in 1:M
        lower_bound_samples[i], x_samples[i] = Q(demand_matrix[i,:])
        x_mean += x_samples[i]
        lower_bound += lower_bound_samples[i]
    end
    std_V[index, 1] = std(x_samples)
    std_V[index, 2] = std(lower_bound_samples)
    LB_V[index] = lower_bound/M
    Mean_V[index] = x_mean/M
end

@show LB_V 
@show Mean_V
@show std_V

y = collect(1:1:20)
quantiles = zeros(20, 2)

for i in 1:length(N_amounts)
    normal_x = Normal(Mean_V[i], std_V[i,1])
    normal_LB = Normal(LB_V[i], std_V[i,2])
    quantiles[i,1], quantiles[i,2] = quantile(normal_x, 0.05), quantile(normal_x, 0.95)
    # @show quantile(normal_LB, 0.025), quantile(normal_LB, 0.975)
end

labels = permutedims(["N = $(N_amounts[i])" for i in 1:length(N_amounts)])
#plot horizontal lines for the confidence intervals of the mean
Plots.plot(quantiles', [y;;y]', palette = :darktest, linewidth = 2, legend = :topright, xlabel = "X", ylabel = "Number of samples divided by 50", title = "Confidence intervals of the mean", size = (800, 600), label = labels)

## Computing upper bound

# Obtaining 10 candidate solutions with 10 batches of 1000 samples
demand_matrix = rand(cont_unif, M, 1000)
x̂_vector = zeros(M)
for i in 1:M
    obj, x = Q(demand_matrix[i,:])
    x̂_vector[i] = x
end

# Use the 10 candidate solutions in 1000 out of sample scenarios to compute the upper bound
K = 1000
out_of_sample = rand(cont_unif, M, K)
UB_V = zeros(M)
for (i,x) in enumerate(x̂_vector)
    UB_V[i] = 10*x + (1/K)*sum(Second_Stage(x, out_of_sample[i,j]) for j in 1:K)
end

UB_V
LB_V