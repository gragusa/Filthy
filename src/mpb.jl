using MathProgBase
function MathProgBase.initialize(d::SSMOptim, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(d::SSMOptim) = [:Grad, :Jac]

function MathProgBase.eval_f(d::SSMOptim, x)  
    ssm_obj_fun(d, x)
end

function MathProgBase.eval_g(d::SSMOptim, g, x)    
end

function MathProgBase.eval_grad_f(d::SSMOptim, grad_f, x)
    ForwardDiff.gradient!(grad_f, y->ssm_obj_fun(d, y), x, d.d.jcfg, Val{false}())
end

MathProgBase.jac_structure(d::SSMOptim) = Int[],Int[]
# lower triangle only
MathProgBase.hesslag_structure(d::SSMOptim) = Int[]


function MathProgBase.eval_jac_g(d::SSMOptim, J, x)
end

function MathProgBase.eval_hesslag(d::SSMOptim, H, x, σ, μ)
end
