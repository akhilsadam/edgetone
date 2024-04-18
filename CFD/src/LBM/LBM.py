import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate flow past cylinder
for an isothermal fluid


Modified from original at `https://github.com/pmocz/latticeboltzmann-python` to serve as LBM solver w POD for edgetone?
Akhil Sadam (2024)

"""

export = "p.npy"

cylinder = lambda X, Y, nx, ny : (X - nx/8)**2 + (Y - ny/2)**2 < (ny/4)**2
edgeA = lambda X, Y, nx, ny : (Y < np.minimum(0.1 * (X - nx/8),5) + ny/2) * (Y > ny/2) * (X < 3*nx/4)

# jet = lambda Y, ny : 0.005 * np.clip(10 - (Y - ny/2)**2,0,10) # less than 1

flat = lambda Y, ny : 0.01  + 0 * Y # ~ 3m/s....

def LBM(inlet_profile, boundary, nx = 200, ny = 100, nt = 1000, rho_avg = 100, nu = 0.4):
    """ Lattice Boltzmann Simulation """
    
    # Simulation parameters
    # nx        # resolution x-dir
    # ny        # resolution y-dir
    # nt        # number of timesteps
    # rho_avg   # average density
    # nu        # kinematic viscosity
    
    ### unit scale: speed of sound is 1, dt is 1 ###
    
    tau = nu + 0.5
    
    # tau       # collision timescale
    
    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
    # downwind (reverse) propagation is the last three indices, as seen in cxs!
    
    
    # Initial Conditions
    F = np.ones((ny,nx,NL)) #* rho_avg / NL
    np.random.seed(42)
    F += 0.01*np.random.randn(ny,nx,NL)
    X, Y = np.meshgrid(range(nx), range(ny))
    ys = np.arange(ny)
    
    # F[:,:,3] += 2* (1+0.2*np.cos(2*np.pi*X/nx*4)) # starting x-velocity
    
    rho = np.sum(F,2)
    # for i in idxs:
    #     F[:,:,i] *= rho_avg / rho
    
    #opt
    F = F * ((rho_avg/rho)[:,:,None])
    
    # Cylinder boundary
    X, Y = np.meshgrid(range(nx), range(ny))
    cylinder = boundary(X, Y, nx, ny)
    
    # walls
    buf=1
    inlet = X==buf
    outlet = X>=nx-buf
    walls = (Y==0) + (Y==ny-1)
    
    
    # "pressure" data matrix 
    p = np.empty((nt, ny, nx), dtype=np.float32)
    
    # save outlet
    # outlet = F[:,-1,:]
    Feq = np.zeros(F.shape)
    
    # Simulation Main Loop
    for it in range(nt):
        print(it)

        # transparent (freestream) boundaries
        out_free = F[outlet,:]

        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
        out_free[:,[0,1,2,3,4,5]] = F[outlet,:][:,[0,1,2,3,4,5]]
        
        F[outlet,:] = out_free
        
        # Set reflective boundaries
        reflect_cyl = F[cylinder,:]
        reflect_cyl = reflect_cyl[:,[0,5,6,7,8,1,2,3,4]]      
                
        # inlet condition
        F[inlet,3] = inlet_profile(ys, ny) # only along axis 3
        
        # Calculate fluid variables
        rho = np.sum(F,2)
        ux  = np.sum(F*cxs,2) / rho
        uy  = np.sum(F*cys,2) / rho
        
        # Apply Collision
        # for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        #     Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
        
        #opt
        uxf = ux[:,:,None]
        uyf = uy[:,:,None]
        cxf = cxs[None,None,:]
        cyf = cys[None,None,:]
        Feq = rho[:,:,None] * weights[None,None,:] * ( 1 + 3*(cxf*uxf+cyf*uyf)  + 9*(cxf*uxf+cyf*uyf)**2/2 - 3*(uxf**2+uyf**2)/2 )
        
        
        F += -(1.0/tau) * (F - Feq)
        
        # assert (ux<1).all(),f"{it}"
        
        # Apply boundaries 
        F[cylinder,:] = reflect_cyl
  
          
        # output pitot pressure / mic
        ux[cylinder] = 0
        uy[cylinder] = 0
        p[it,:,:] = rho * (ux ** 2 + uy ** 2) # omitting 1/2 factor

        # # plot in real time - color 1/2 particles blue, other half red
        # if (plotRealTime and (it % 10) == 0) or (it == nt-1):
        # 	plt.cla()
        # 	ux[cylinder] = 0
        # 	uy[cylinder] = 0
        # 	vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        # 	vorticity[cylinder] = np.nan
        # 	vorticity = np.ma.array(vorticity, mask=cylinder)
        # 	plt.imshow(vorticity, cmap='bwr')
        # 	plt.imshow(~cylinder, cmap='gray', alpha=0.3)
        # 	plt.clim(-.1, .1)
        # 	ax = plt.gca()
        # 	ax.invert_yaxis()
        # 	ax.get_xaxis().set_visible(False)
        # 	ax.get_yaxis().set_visible(False)	
        # 	ax.set_aspect('equal')	
        # 	plt.pause(0.001)
            
    
    # # Save figure
    # plt.savefig('latticeboltzmann.png',dpi=240)
    # plt.show()
        
    return p

def anim(x):
    fig, ax = plt.subplots(figsize=(5,2))
    def make_frame(f):
        ax.clear()
        im = ax.imshow(f, cmap="coolwarm")
        plt.tight_layout()
        return mplfig_to_npimage(fig)
    animation = ImageSequenceClip(list(make_frame(f) for f in x), fps=10)
    plt.close()
    return animation

if __name__== "__main__":
    from moviepy.editor import ImageSequenceClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    anim(LBM(flat,edgeA)).write_videofile("movie.mp4",fps=15) # np.save(export,p) # np.load(export)
