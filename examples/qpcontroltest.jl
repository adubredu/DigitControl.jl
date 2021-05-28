using QPControl
using RigidBodyDynamics
using RigidBodyDynamics.PDControl
using RigidBodyDynamics.Contact
using StaticArrays
using DigitRobot
using BenchmarkTools
using Test
using RigidBodySim

# load URDF
mechanism = DigitRobot.mechanism()
remove_fixed_tree_joints!(mechanism);

# add environment
rootframe = root_frame(mechanism)
ground = HalfSpace3D(Point3D(rootframe, 0., 0., 0.), FreeVector3D(rootframe, 0., 0., 1.))
add_environment_primitive!(mechanism, ground);

# create optimizer
using MathOptInterface
using OSQP
using OSQP.MathOptInterfaceOSQP: OSQPSettings
const MOI = MathOptInterface
optimizer = OSQP.Optimizer()
MOI.set(optimizer, OSQPSettings.Verbose(), false)
MOI.set(optimizer, OSQPSettings.EpsAbs(), 1e-5)
MOI.set(optimizer, OSQPSettings.EpsRel(), 1e-5)
MOI.set(optimizer, OSQPSettings.MaxIter(), 5000)
MOI.set(optimizer, OSQPSettings.AdaptiveRhoInterval(), 25) # required for deterministic behavior

# create low level controller
const num_basis_vectors = 4
lowlevel = MomentumBasedController{num_basis_vectors}(mechanism, optimizer,
    floatingjoint = findjoint(mechanism, "torso_to_world"));
for body in bodies(mechanism)
    for point in RigidBodyDynamics.contact_points(body)
        position = location(point)
        normal = FreeVector3D(default_frame(body), 0.0, 0.0, 1.0)
        μ = point.model.friction.μ
        contact = addcontact!(lowlevel, body, position, normal, μ)
        contact.maxnormalforce[] = 1e6 # TODO
        contact.weight[] = 1e-3
    end
end

# state initialization
function initialize!(digitstate::MechanismState)
    mechanism = digitstate.mechanism
    zero!(digitstate)

    hip_joint_left = findjoint(mechanism, "hip_abduction_left")
    hip_joint_right = findjoint(mechanism, "hip_abduction_right")
    toe_joint_left = findjoint(mechanism, "toe_pitch_joint_left")
    toe_joint_right = findjoint(mechanism, "toe_pitch_joint_right")

    set_configuration!(digitstate, hip_joint_left, 0.337)
    set_configuration!(digitstate, hip_joint_right, -0.337)
    set_configuration!(digitstate, toe_joint_left, -0.126)
    set_configuration!(digitstate, toe_joint_right, 0.126)

    floatingjoint = first(out_joints(root_body(mechanism), mechanism))
    set_configuration!(digitstate, floatingjoint, [1; 0; 0; 0; 0; 0; 0.85])
    digitstate
end


# create standing controller
feet = findbody.(Ref(mechanism), ["left_toe_roll", "right_toe_roll"])
pelvis = findbody(mechanism, "torso")
nominalstate = MechanismState(mechanism)
initialize!(nominalstate)
controller = StandingController(lowlevel, feet, pelvis, nominalstate);

state = MechanismState(mechanism)
initialize!(state)
τ = similar(velocity(state));
# benchresult = @benchmark $controller($τ, 0.0, $state)
# @show benchresult.allocs
# @test benchresult.allocs <= 24
# benchresult

# set up visualizer
using MeshCat
using MeshCatMechanisms
if !@isdefined(vis) || !any(isopen, vis.core.scope.pool.connections)
    vis = Visualizer()[:digit]
    visuals = URDFVisuals(DigitRobot.urdfpath(); package_path = [DigitRobot.packagepath()])
    mvis = MechanismVisualizer(mechanism, visuals, vis)
    set_configuration!(mvis, configuration(nominalstate))
    open(mvis)
    wait(mvis)
end

state = MechanismState(mechanism)
initialize!(state)
Δt = 1 / 500
pcontroller = PeriodicController(similar(velocity(state)), Δt, controller)
# TODO: add damping
dynamics = Dynamics(mechanism, pcontroller)
problem = ODEProblem(dynamics, state, (0., 10.))

sol = solve(problem, Tsit5(), abs_tol = 1e-8, dt = 1e-6)
@time sol = solve(problem, Tsit5(), abs_tol = 1e-8, dt = 1e-6)
@test sol.retcode == :Success
copyto!(state, last(sol.u))
@test norm(velocity(state)) ≈ 0 atol=1e-8
@test center_of_mass(state).v[3] > 1

setanimation!(mvis, sol)