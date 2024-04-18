import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import matplotlib.pyplot as plt

from tqdm import tqdm


"""
CFD solver w POD for edgetone?
Akhil Sadam (2024)

"""

export = "pv.npy"


flat = lambda Y, ny : 1.0 
cylinder = lambda X, Y, nx, ny : (X - nx/8)**2 + (Y - ny/2)**2 < (ny/4)**2

def edge(st=0.1, end=0.4, rise=0.2, rate=0.5):
    return lambda X, Y, nx, ny : (Y < np.minimum(rate * (X - st*nx), rise*ny) + ny/2) * (Y > ny/2) * (X < end*nx)

def jet(velocity=1.0,width=0.1,center=0.5):
    return lambda Y, ny : velocity * ((Y < (ny * (center+width))) * (Y > (ny * (center-width))))

def FD(inlet_profile, boundary, hx = 90, hy = 30, nt = 1000, dt=0.1, nu = 0.0001, pipe_length = 1.0):
    """
    Solves the incompressible Navier Stokes equations using "Stable Fluids" by Jos Stam
    in a closed box with a forcing that creates a bloom. 


    Momentum:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u + f

    Incompressibility:  ∇ ⋅ u = 0


    u:  Velocity (2d vector)
    p:  Pressure
    f:  Forcing
    ν:  Kinematic Viscosity
    ρ:  Density
    t:  Time
    ∇:  Nabla operator (defining nonlinear convection, gradient and divergence)
    ∇²: Laplace Operator


    ----

    A closed box


                                        u = 0
                                        v = 0 

            1 +-------------------------------------------------+
                |                                                 |
                |             *                      *            |
                |          *           *    *    *                |
            0.8 |                                                 |
                |                                 *               |
                |     *       *                                   |
                |                      *     *                    |
            0.6 |                                            *    |
    u = 0       |      *                             *            |   u = 0
    v = 0       |                             *                   |   v = 0
                |                     *                           |
                |           *                *         *          |
            0.4 |                                                 |
                |                                                 |
                |      *            *             *               |
                |           *                             *       |
            0.2 |                       *           *             |
                |                               *                 |
                |  *          *      *                 *       *  |
                |                            *                    |
            0 +-------------------------------------------------+
                0        0.2       0.4       0.6       0.8        1

                                        u = 0
                                        v = 0

    * Homogeneous Dirichlet Boundary Condition

    ----- 

    Forcing Function

            1 +-------------------------------------------------+
                |                                                 |
                |             *                      *            |
                |          *           *    *    *                |
            0.8 |                                                 |
                |                                 *               |
                |     *       *                                   |
                |                      *     *                    |
            0.6 |                                            *    |
                |      *                             *            |
                |                             *                   | 
                |                     *                           |
                |           *                *         *          |
            0.4 |                                                 |
                |                                                 |
                |      *           ^ ^ ^ ^ ^ ^                    |
                |           *      | | | | | |                 *  |
            0.2 |                  | | | | | |     *           *  |
                |                  | | | | | |             *      |
                |  *          *      *                 *       *  |
                |                            *                    |
            0 +-------------------------------------------------+
                0        0.2       0.4       0.6       0.8        1

    -> Upwards pointing force in the lower center of the domain.


    -----

    Solution Strategy:

    -> Start with zero velocity everywhere: u = [0, 0]

    1. Add forces

        w₁ = u + Δt f

    2. Convect by self-advection (set the value at the current
    location to be the value at the position backtraced
    on the streamline.) -> unconditionally stable

        w₂ = w₁(p(x, −Δt))

    3. Diffuse implicitly (Solve a linear system matrix-free
    by Conjugate Gradient) -> unconditionally stable

        (I − ν Δt ∇²)w₃ = w₂

    4.1 Compute a pressure correction (Solve a linear system
        matrix-free by Conjugate gradient)

        ∇² p = ∇ ⋅ w₃

    4.2 Correct velocities to be incompressible

        w₄ = w₃ − ∇p

    5. Advance to next time step

        u = w₄ 


    -> The Boundary Conditions are prescribed indirectly using
    the discrete differential operators.

    -----

    The solver is unconditionally stable, hence all parameters can be
    chosen arbitrarily. Still, be careful that too high timesteps can
    make the advection step highly incorrect.
    """
    N_TIME_STEPS = nt
    TIME_STEP_LENGTH = dt

    KINEMATIC_VISCOSITY = nu

    MAX_ITER_CG = 1000

    # def forcing_function(time, point):
    #     time_decay = np.maximum(
    #         2.0 - 0.5 * time,
    #         0.0,
    #     )

    #     forced_value = (
    #         time_decay
    #         *
    #         np.where(
    #             (
    #                 (point[0] > 0.4)
    #                 &
    #                 (point[0] < 0.6)
    #                 &
    #                 (point[1] > 0.1)
    #                 &
    #                 (point[1] < 0.3)
    #             ),
    #             np.array([0.0, 1.0]),
    #             np.array([0.0, 0.0]),
    #         )
    #     )

    #     return forced_value

    nx = hx + 1
    ny = hy + 1
    
    element_length = pipe_length / hx
    scalar_shape = (nx, ny)
    scalar_dof = nx * ny
    vector_shape = (nx, ny, 2)
    vector_dof = nx * ny * 2
    
    DOMAIN_SIZE_X = pipe_length
    DOMAIN_SIZE_Y = element_length * hy
    
    _output = np.empty((N_TIME_STEPS, nx, ny, 3), dtype=np.float32)

    x = np.linspace(0.0, DOMAIN_SIZE_X, nx)
    y = np.linspace(0.0, DOMAIN_SIZE_Y, ny)

    # Using "ij" indexing makes the differential operators more logical. Take
    # care when plotting.
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    cylinder = boundary(X, Y, DOMAIN_SIZE_X, DOMAIN_SIZE_Y)
    
    # walls
    inlet = X<x[1]
    # pre_inlet = X==x[1]
    
    pre_outlet = X==x[nx-2]
    outlet = X==x[nx-1]
    
    pre_wall_up = Y==y[1]
    wall_up = Y==y[0]
    
    pre_wall_dn = Y==y[ny-2]
    wall_dn = Y==y[ny-1]

    coordinates = np.concatenate(
        (
            X[..., np.newaxis],
            Y[..., np.newaxis],
        ),
        axis=-1,
    )

    # forcing_function_vectorized = np.vectorize(
    #     pyfunc=forcing_function,
    #     signature="(),(d)->(d)",
    # )

    def partial_derivative_x(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[2:  , 1:-1]
                -
                field[0:-2, 1:-1]
            ) / (
                2 * element_length
            )
        )

        return diff

    def partial_derivative_y(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[1:-1, 2:  ]
                -
                field[1:-1, 0:-2]
            ) / (
                2 * element_length
            )
        )

        return diff

    def laplace(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[0:-2, 1:-1]
                +
                field[1:-1, 0:-2]
                - 4 *
                field[1:-1, 1:-1]
                +
                field[2:  , 1:-1]
                +
                field[1:-1, 2:  ]
            ) / (
                element_length**2
            )
        )

        return diff
    
    def divergence(vector_field):
        divergence_applied = (
            partial_derivative_x(vector_field[..., 0])
            +
            partial_derivative_y(vector_field[..., 1])
        )

        return divergence_applied
    
    def gradient(field):
        gradient_applied = np.concatenate(
            (
                partial_derivative_x(field)[..., np.newaxis],
                partial_derivative_y(field)[..., np.newaxis],
            ),
            axis=-1,
        )

        return gradient_applied
    
    def curl_2d(vector_field):
        curl_applied = (
            partial_derivative_x(vector_field[..., 1])
            -
            partial_derivative_y(vector_field[..., 0])
        )

        return curl_applied

    def advect(field, vector_field):
        backtraced_positions = np.clip(
            (
                coordinates
                -
                TIME_STEP_LENGTH
                *
                vector_field
            ),
            0.0,
            np.array([DOMAIN_SIZE_X, DOMAIN_SIZE_Y])[None,None,:],
        )

        advected_field = interpolate.interpn(
            points=(x, y),
            values=field,
            xi=backtraced_positions,
        )

        return advected_field
    
    def diffusion_operator(vector_field_flattened):
        vector_field = vector_field_flattened.reshape(vector_shape)

        diffusion_applied = (
            vector_field
            -
            KINEMATIC_VISCOSITY
            *
            TIME_STEP_LENGTH
            *
            laplace(vector_field)
        )

        return diffusion_applied.flatten()
    
    def poisson_operator(field_flattened):
        field = field_flattened.reshape(scalar_shape)

        poisson_applied = laplace(field)

        return poisson_applied.flatten()

    velocities_prev = np.zeros(vector_shape)
    
    time_current = 0.0
    for i in tqdm(range(N_TIME_STEPS)):
        time_current += TIME_STEP_LENGTH

        # forces = forcing_function_vectorized(
        #     time_current,
        #     coordinates,
        # )

        # (1) Apply Forces
        velocities_forces_applied = (
            velocities_prev
            # +
            # TIME_STEP_LENGTH
            # *
            # forces
        )

        # (2) Nonlinear convection (=self-advection)
        velocities_advected = advect(
            field=velocities_forces_applied,
            vector_field=velocities_forces_applied,
        )

        # (3) Diffuse
        velocities_diffused = splinalg.cg(
            A=splinalg.LinearOperator(
                shape=(vector_dof, vector_dof),
                matvec=diffusion_operator,
            ),
            b=velocities_advected.flatten(),
            maxiter=MAX_ITER_CG,
        )[0].reshape(vector_shape)
        
                        
        # (3.5) Boundary Conditions
        # inlet
        velocities_diffused[inlet, 0] = inlet_profile(y,DOMAIN_SIZE_Y)
        velocities_diffused[inlet, 1] = 0
        
        # walls & outlet (both outflow)
        velocities_diffused[wall_up, :] = velocities_diffused[pre_wall_up, :]
        velocities_diffused[wall_dn, :] = velocities_diffused[pre_wall_dn, :]
        # velocities_diffused[wall_up, :] = 0
        # velocities_diffused[wall_dn, :] = 0
        velocities_diffused[outlet, :] = velocities_diffused[pre_outlet, :]
        
        # (3.6) Boundary Conditions for edge/pipe
        velocities_diffused[cylinder] = 0
                    

        # (4.1) Compute a pressure correction
        pressure = splinalg.cg(
            A=splinalg.LinearOperator(
                shape=(scalar_dof, scalar_dof),
                matvec=poisson_operator,
            ),
            b=divergence(velocities_diffused).flatten(),
            maxiter=MAX_ITER_CG,
        )[0].reshape(scalar_shape)

        # (4.2) Correct the velocities to be incompressible
        velocities_projected = (
            velocities_diffused
            -
            gradient(pressure)
        )

        # Advance to next time step
        velocities_prev = velocities_projected


        # output
        _output[i,...,1:] = velocities_prev
        _output[i,...,0] = pressure
        
    return _output
    


def anim(x):
    fig, ax = plt.subplots(figsize=(5,5))
    def make_frame(f):
        ax.clear()
        im = ax.imshow(f.T, cmap="coolwarm")
        plt.tight_layout()
        return mplfig_to_npimage(fig)
    animation = ImageSequenceClip(list(make_frame(f) for f in x), fps=20)
    plt.close()
    return animation

if __name__== "__main__":
    
    pv = FD(jet(),edge())
    np.save(export, pv)
    
    from moviepy.editor import ImageSequenceClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    anim(pv[...,0]).write_videofile("pressure.mp4") # np.save(export,p) # np.load(export)
