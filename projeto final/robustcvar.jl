using JuMP, LinearAlgebra, Statistics, DataFrames, AlphaVantage, CSV, HiGHS

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
    start_date::Int64, n_steps::Int64, initial_funds)
    if start_date > 1000
        println("Choose a starting date before 1000 days ago")
    end
    #Reading returns data
    returns_df = CSV.read(joinpath(@__DIR__, "./returns.csv"), DataFrame)
    #Converting to matrix
    returns = Matrix{Float64}(returns_df)
    returns = returns[(start_date - window_size - n_steps):start_date, :]

    allocation = robust_cvar_allocation(returns[n_steps:(window_size + n_steps), :],
        risk_threshold, 0.95, 1, initial_funds, 1, 1)
    
    portfolio_returns_future = zeros(n_steps)
    #Start rolling window backtest
    for i in 1:n_steps-1
        portfolio_returns_future[i] = allocation ⋅ (1 .+ returns[n_steps - i, :])
        allocation = robust_cvar_allocation(returns[(n_steps - i):(window_size + n_steps - i), :],
         risk_threshold, 0.95, 1, portfolio_returns_future[i], 1, 1)
        println(allocation)
         if allocation === nothing
            println("No allocation found for step ", i)
            return portfolio_returns_future
        end
    end
    return portfolio_returns_future
end

function robust_cvar_allocation(returns, T_r, α, Γ, funds, γ, former_allocation)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    n_assets = size(returns, 2)
    n_scenarios = size(returns, 1)
    r̄ = mean(returns, dims=1)
    σ = std(returns, dims=1)
    r̄ = r̄[1,:]
    σ = σ[1,:]
    @variables(m,
    begin
        P
        x[1:n_assets] ≥ 0
        λ[1:n_assets] ≥ 0
        Δ_p[1:n_assets] ≥ 0
        Δ_n[1:n_assets] ≤ 0
        θ[1:n_scenarios] ≥ 0
        l
        β
    end)
    @constraint(m, con1, sum(x) == funds)
    @constraint(m, con2, P ≤ r̄ ⋅ x)
    @constraint(m, con3[s=1:n_scenarios], θ[s] ≥ -returns[s, :] ⋅ x - l)
    @constraint(m, con4, l + sum(θ)*(1/n_scenarios)/(1-α) ≤ T_r*funds)
    @constraint(m, con5[n=1:n_assets], -Δ_p[n] + Δ_n[n] ≥ -x[n])
    @constraint(m, con6[n=1:n_assets], -σ[n]*(Δ_p[n] + Δ_n[n]) - β - λ[n] ≥ 0)
    @constraint(m, con7[n=1:n_assets], -Γ*β - sum(λ) + sum(r̄[n]*(Δ_p[n] - Δ_n[n])) ≥ P)
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
allocation = robust_cvar_allocation(mat, 0.039, 0.95, 1, 10e6, 0.01, 0.01)

cvar(allocation, mat, 0.95)


returns_vec = backtest(100, 0.01, 0.1, 300, 50, 10e6)