using JuMP, LinearAlgebra, Statistics, DataFrames, AlphaVantage, CSV, HiGHS, StatsPlots

plotlyjs()

function cvar(x, returns, α)
    n_scenarios = size(returns, 1)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, θ[1:n_scenarios] >= 0)
    @variable(m, l)
    @constraint(m, con3[s=1:n_scenarios], θ[s] ≥ -returns[s, :] ⋅ x - l)
    @objective(m, Min, l + sum(θ)*(1/n_scenarios)/(1-α))
    optimize!(m)
    return objective_value(m), value.(l)
end

function backtest(window_size::Int64, brokerage_rate::Float64, risk_threshold::Float64,
    start_date::Int64, n_steps::Int64, initial_funds, Γ::Vector{Float64}, benchmark::Bool)
    
    if start_date > 1000
        println("Choose a starting date before 1000 days ago")
    end
    
    #Reading returns data
    returns_df = CSV.read(joinpath(@__DIR__, "./returns.csv"), DataFrame)
    
    #Converting to matrix
    returns = Matrix{Float64}(returns_df)
    returns = returns[(start_date - window_size - n_steps):start_date, :]

    #Naming dataframe columns
    cols = ["Γ = $robustness" for robustness in Γ]

    if benchmark
        push!(cols, "Benchmark")
    end

    results_matrix = zeros(n_steps, length(cols))

    uniform_allocation = [initial_funds/length(returns[1, :]) for i in 1:length(returns[1, :])]
    
    for (index, robustness) in enumerate(Γ)
    
        window = returns[n_steps:(window_size + n_steps), :]

        allocation = robust_cvar_allocation(window, risk_threshold, 0.95, robustness, initial_funds, brokerage_rate, uniform_allocation)
        
        portfolio_returns_future = zeros(n_steps)

        #Start rolling window backtest
        for i in 1:n_steps-1
            portfolio_returns_future[i] = allocation ⋅ (1 .+ returns[n_steps - i, :])
            allocation = robust_cvar_allocation(returns[(n_steps - i):(window_size + n_steps - i), :],
            risk_threshold, 0.95, robustness, portfolio_returns_future[i], brokerage_rate, allocation)
            if allocation === nothing
                println("No allocation found for step ", i)
                break
            end
        end

        results_matrix[:, index] = portfolio_returns_future
    
    end

    if benchmark
        
        allocation = copy(uniform_allocation)
        portfolio_returns_future = zeros(n_steps)

        for i in 1:n_steps-1
            portfolio_returns_future[i] = allocation ⋅ (1 .+ returns[n_steps - i, :])
            allocation = allocation .* (1 .+ returns[n_steps - i, :])
        end
        results_matrix[:, end] = portfolio_returns_future
    
    end

    return DataFrame(results_matrix, cols)

end

function robust_cvar_allocation(returns, T_r, α, Γ, funds, γ, former_allocation)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    n_assets = size(returns, 2)
    n_scenarios = size(returns, 1)
    r̄ = mean(returns, dims=1)
    σ = std(returns, dims=1)
    r̄ = r̄[1,:] .+ ones(n_assets)
    σ = σ[1,:]
    @variables(m,
    begin
        P ≥ 0
        x[1:n_assets] ≥ 0
        λ[1:n_assets] ≤ 0
        Δ_p[1:n_assets] ≥ 0
        Δ_n[1:n_assets] ≥ 0
        θ[1:n_scenarios] ≥ 0
        l
        β ≤ 0
        c[1:n_assets] ≥ 0
        v[1:n_assets] ≥ 0
    end)
    @constraint(m, con0, sum(v) == sum(c) + γ*(sum(c .+ v)))
    @constraint(m, con1[n=1:n_assets], x[n] == former_allocation[n] + c[n] - v[n])
    @constraint(m, con3[s=1:n_scenarios], θ[s] ≥ -returns[s, :] ⋅ x - l)
    @constraint(m, con4, l + sum(θ)*(1/n_scenarios)/(1-α) ≤ T_r*sum(former_allocation))
    @constraint(m, con5[n=1:n_assets], -Δ_p[n] + Δ_n[n] ≥ -x[n])
    @constraint(m, con6[n=1:n_assets], -σ[n]*(Δ_p[n] + Δ_n[n]) - β - λ[n] ≥ 0)
    # @constraint(m, con7, Γ*β + sum(λ) + sum((1 .+ r̄) .* (Δ_p .- Δ_n)) ≥ P)
    @constraint(m, con7, Γ*β + sum(λ) + sum(r̄ .* (Δ_p .- Δ_n)) ≥ P)
    @objective(m, Max, P)
    optimize!(m)
    #Check if the problem is feasible
    if termination_status(m) != MOI.OPTIMAL
        println("The problem is not feasible")
        return nothing
    end
    return value.(x)

end

#Reading returns data
returns_df = CSV.read(joinpath(@__DIR__, "./returns.csv"), DataFrame)
mat = Matrix{Float64}(returns_df)
initial_allocation = [10/length(mat[1, :]) for i in 1:length(mat[1, :])]

allocation = robust_cvar_allocation(mat, 0.05, 0.95, 2, 10e6, 0.01, initial_allocation)

cvar(allocation, mat, 0.95)


final_df = backtest(100, 0.01, 100.0, 672, 100, 10, [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true)

CSV.write("resultado.csv", final_df)

delete!(final_df, [99])

@df final_df plot(cols(), legend = :outerbottomleft)

@show returns_df