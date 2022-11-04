using JuMP, LinearAlgebra, Statistics, DataFrames, AlphaVantage, CSV, HiGHS


function cvar(x)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, l)
    @variable(m, θ[1:N_scenarios])
    @constraint(m, con1[s=1:N_scenarios], θ[s] ≥ 0)
    @constraint(m, con2[s=1:N_scenarios], θ[s] ≥ (-r[s, :] ⋅ x) - l)
    @objective(m, Min, l + sum(p[s]*θ[s] for s=1:N_scenarios)/(1-α))
    optimize!(m)
    return objective_value(m)
end

function robust_cvar_allocation(returns, T_r, T_p, α, Γ)
    m = Model(HiGHS.Optimizer)
    n_assets = size(returns, 2)
    n_scenarios = size(returns, 1)
    r̄ = mean(returns, dims=1)
    σ = std(returns, dims=1)
    r̄ = r̄[1,:]
    σ = σ[1,:]
    @variables(m,
    begin
        ω ≥ 0
        x[1:n_assets] ≥ 0
        λ[1:n_assets] ≥ 0
        Δ_p[1:n_assets] ≥ 0
        Δ_n[1:n_assets] ≥ 0
        θ[1:n_scenarios] ≥ 0
        l
        β ≥ 0
    end)
    @constraint(m, con1, sum(x) == 1)
    @constraint(m, con2, ω ≤ r̄ ⋅ x)
    @constraint(m, con3[s=1:n_scenarios], θ[s] ≥ -returns[s, :] ⋅ x - l)
    @constraint(m, con4[s=1:n_scenarios], l + sum(θ[s])*(1/n_scenarios)/(1-α) ≤ T_r)
    @constraint(m, con5[n=1:n_assets], Δ_p[n] - Δ_n[n] ≥ x[n])
    @constraint(m, con6[n=1:n_assets], σ[n]*(Δ_p[n] + Δ_n[n]) - β - λ[n] ≥ 0)
    @constraint(m, con7[n=1:n_assets], -Γ*β - sum(λ) + sum(r̄[n]*(Δ_p[n] - Δ_n[n])) ≥ T_p)
    @objective(m, Max, ω)
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

#Converting to matrix
returns = Matrix{Float64}(returns_df)

@show returns[1:5, 1:5]

allocation = robust_cvar_allocation(returns, -5, 0.0, 0.95, 1)

size(returns, 2)

r̄ = mean(returns, dims=1)

# Convert to vector
r̄ = r̄[1, :]

allocation ⋅ r̄

cvar(allocation)