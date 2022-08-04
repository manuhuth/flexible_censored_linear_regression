using Pkg

#Pkg.add(["PrettyTables"])

using Random
using Base.Threads
using BenchmarkTools
using XLSX
using DataFrames
using Distributions
using ForwardDiff
using Optim
using LinearAlgebra
using Plots
using PrettyTables




#---------load data--------------------------------------------------
function load_test_data(path)
    return DataFrame(XLSX.readtable(path, 1, infer_eltypes=true)...)
end

function extract_dependent(formula)
    y = strip(split.(formula, "~")[1])
    return y
end

function extract_X_names(formula)
    X = map(strip, split.(split.(formula, "~")[2], "+"))

    functional_covariates = X[findall( x -> occursin("f(", x), X)]
    scalar_covariates = filter!(e->e∉functional_covariates,X)

    return chop.(functional_covariates, head=2, tail=1), scalar_covariates
end

function get_censored_likelihood_contribution_i(y, mean, sigma, left, right)
    if (left + right) == 0
        return pdf(Normal(mean, sigma), y)
    elseif left == 1
        return  cdf(Normal(mean, sigma), y)
    elseif right == 1
        return cdf(Normal(mean, sigma), -y)
    end
end

function compute_log_likelihood(y, X, theta, left, right)
    sum = 0.0
    for i in 1:length(y)
        sum = sum - log(get_censored_likelihood_contribution_i(y[i], X[i, :]'*theta[1:(length(theta)-1)], last(theta), left[i], right[i]))
    end
    return sum
end

function compute_inverse_matrix(M)

  return inv(M)
end

function compute_hessian(f, x)
  return ForwardDiff.hessian(f, x)
end

function censored_regression_flexible(formula, data, left_censored=nothing,
     right_censored=nothing, starting_values=nothing, optimizer=Fminbox(LBFGS()))
    x_names = extract_X_names(formula)[2]
    y_name = extract_dependent(formula)
    X = Matrix(data[:, x_names])
    y = Array(data[:, y_name])

    if isnothing(left_censored)
        left = zeros(length(y))
    else
        left = data[:, left_censored]
    end

    if isnothing(right_censored)
        right = zeros(length(y))
    else
        right = data[:, right_censored]
    end

    if isnothing(starting_values)
        starting_beta = inv(X' * X) *X' *y
        starting_sigma = sum( (y .- X*starting_beta).^2 ) / (length(y) - length(x_names))
        starting_values = vcat(starting_beta, [starting_sigma])
    end


    lower = vcat(repeat([-Inf], length(x_names) ), [0])
    upper = vcat(repeat([Inf], length(x_names) ), [Inf])


    fn = OnceDifferentiable(theta -> compute_log_likelihood(y, X, theta, left, right),
                                    starting_values, autodiff=:forward
                                    )

    fit = optimize(fn, lower, upper, starting_values,
            optimizer)#, Optim.Options(show_every = 1, show_trace = true))

    parameter = Optim.minimizer(fit)

    vcv = compute_inverse_matrix(compute_hessian(theta -> compute_log_likelihood(y, X, theta, left, right), parameter))
    se = diag(vcv).^0.5
    t_values = parameter ./ se

    p_values = 2 .* (1 .- cdf.(Normal(), abs.(t_values)))


    data_frame_out = DataFrame(Parameter= vcat(x_names, ["sigma"]), Estimates=round.(parameter, digits=4),
                                Se=round.(se, digits=4), t=round.(t_values, digits=4), p=round.(p_values, digits=4),
                                Significance=get_significance_stars(p_values))

    output = Dict("fit" => fit, "parameter" => parameter,
                "vcv" => compute_inverse_matrix(compute_hessian(theta -> compute_log_likelihood(y, X, theta, left, right), parameter)),
                "se" => se,
                "t_values" => t_values,
                "p_value" => p_values,
                "x_names" => x_names,
                "prediction" => X * parameter[1:(length(parameter)-1)],
                "X" => X,
                "y" => y,
                "left_cens" => left,
                "right_cens" => right,
                "starting_values" => starting_values,
                "data" => data,
                "table" => data_frame_out
                )
    return output
end

function transform_prediciton!(data, prediction, cens_left, cens_right, cens_predicted)
    for i in 1:nrow(data)
        if data[i, "cens_left"] == 1
            if y[i] >= prediction[i]
                cens_predicted[i] = y[i]
            else
                cens_predicted[i] = prediction[i]
            end
        elseif data[i, "cens_right"] == 1

            if y[i] <= prediction[i]
                cens_predicted[i] = y[i]
            else
                cens_predicted[i] = prediction[i]
            end
        else
            cens_predicted[i] = prediction[i]
        end
    end
    return cens_predicted
end

function get_significance_stars(p_values)
    significance = Array{String}(undef, length(p_values))
    for i in 1:length(p_values)
        if p_values[i] < 0.01
            significance[i] = "***"
        elseif p_values[i] < 0.05
            significance[i] = "**"
        elseif p_values[i] < 0.1
            significance[i] = "*"
        else
            significance[i] = " "
        end
    end

    return significance
end
#-----------do analysis--------------------------------------------------------

global formula = "..."
global fit = censored_regression_flexible(formula)

global cens_predicted = transform_prediciton!(fit["data"], fit["prediction"], "cens_left",  "cens_right", Array{Float64}(undef, length(fit["y"])))

plot(fit["y"], cens_predicted, seriestype= :scatter,
    xlabel="True values", ylabel = "Fitted values",
    )
pretty_table(fit["table"])
