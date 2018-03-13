function MathProgBase.initialize(d::OptimSSM, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(d::OptimSSM) = [:Grad, :Jac]

function MathProgBase.eval_f(d::OptimSSM, x)  
    ## fastloglik 
end

function MathProgBase.eval_g(d::OptimSSM, g, x)
end

function MathProgBase.eval_grad_f(d::OptimSSM, grad_f, x)
    ## Automatica
end

MathProgBase.jac_structure(d::OptimSSM) = [],[]
# lower triangle only
MathProgBase.hesslag_structure(d::OptimSSM) = []


function MathProgBase.eval_jac_g(d::OptimSSM, J, x)
end

function MathProgBase.eval_hesslag(d::OptimSSM, H, x, σ, μ)
end
