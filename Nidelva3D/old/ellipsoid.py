
#%%

#%%

from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
from numpy.linalg import eig, inv


def ls_ellipsoid(xx, yy, zz):
    # finds best fit ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # least squares fit to a 3D-ellipsoid
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
    #
    # Note that sometimes it is expressed as a solution to
    #  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
    # where the last six terms have a factor of 2 in them
    # This is in anticipation of forming a matrix with the polynomial coefficients.
    # Those terms with factors of 2 are all off diagonal elements.  These contribute
    # two terms when multiplied out (symmetric) so would need to be divided by two

    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = xx[:, np.newaxis]
    y = yy[:, np.newaxis]
    z = zz[:, np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x * x, y * y, z * z, x * y, x * z, y * z, x, y, z))
    K = np.ones_like(x)  # column of ones

    # np.hstack performs a loop over all samples and creates
    # a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.

    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ);
    ABC = np.dot(InvJTJ, np.dot(JT, K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa = np.append(ABC, -1)

    return (eansa)


def polyToParams3D(vec, printMe):
    # gets 3D parameters of an ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # convert the polynomial form of the 3D-ellipsoid to parameters
    # center, axes, and transformation matrix
    # vec is the vector whose elements are the polynomial
    # coefficients A..J
    # returns (center, axes, rotation matrix)

    # Algebraic form: X.T * Amat * X --> polynomial form

    if printMe: print('\npolynomial\n', vec)

    Amat = np.array(
        [
            [vec[0], vec[3] / 2.0, vec[4] / 2.0, vec[6] / 2.0],
            [vec[3] / 2.0, vec[1], vec[5] / 2.0, vec[7] / 2.0],
            [vec[4] / 2.0, vec[5] / 2.0, vec[2], vec[8] / 2.0],
            [vec[6] / 2.0, vec[7] / 2.0, vec[8] / 2.0, vec[9]]
        ])

    if printMe: print('\nAlgebraic form of polynomial\n', Amat)

    # See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A3 = Amat[0:3, 0:3]
    A3inv = inv(A3)
    ofs = vec[6:9] / 2.0
    center = -np.dot(A3inv, ofs)
    if printMe: print('\nCenter at:', center)

    # Center the ellipsoid at the origin
    Tofs = np.eye(4)
    Tofs[3, 0:3] = center
    R = np.dot(Tofs, np.dot(Amat, Tofs.T))
    if printMe: print('\nAlgebraic form translated to center\n', R, '\n')

    R3 = R[0:3, 0:3]
    R3test = R3 / R3[0, 0]
    # print('normed \n',R3test)
    s1 = -R[3, 3]
    R3S = R3 / s1
    (el, ec) = eig(R3S)

    recip = 1.0 / np.abs(el)
    axes = np.sqrt(recip)
    if printMe: print('\nAxes are\n', axes, '\n')

    inve = inv(ec)  # inverse is actually the transpose here
    if printMe: print('\nRotation matrix\n', inve)
    return (center, axes, inve)


# let us assume some definition of x, y and z
dn = 120
s3 = np.sqrt(3)
d = 1
wp = np.array([[120, 0, 0],
               [60, 60*s3, 0],
               [-60, 60*s3, 0],
               [-120, 0, 0],
               [-60, -60*s3, 0],
               [60, -60*s3, 0],

               [120, 0, d],
               [60, 60 * s3, d],
               [-60, 60 * s3, d],
               [-120, 0, d],
               [-60, -60 * s3, d],
               [60, -60 * s3, d],

               [120, 0, -d],
               [60, 60 * s3, -d],
               [-60, 60 * s3, -d],
               [-120, 0, -d],
               [-60, -60 * s3, -d],
               [60, -60 * s3, -d]
               ])

x = wp[:, 0]
y = wp[:, 1]
z = wp[:, 2]

#%%


#%%
# get convex hull
surface = np.stack((x, y, z), axis=-1)
hullV = ConvexHull(surface)
lH = len(hullV.vertices)
hull = np.zeros((lH, 3))
for i in range(len(hullV.vertices)):
    hull[i] = surface[hullV.vertices[i]]
hull = np.transpose(hull)

J = np.stack((hull[0] ** 2, hull[1] ** 2, hull[2] ** 2), axis=-1)
coef = np.linalg.inv(J.T @ J) @ (J.T @ np.ones([len(J), 1]))
print(coef)

#%%

# fit ellipsoid on convex hull
eansa = ls_ellipsoid(hull[0], hull[1], hull[2])  # get ellipsoid polynomial coefficients
# eansa = ls_ellipsoid(x, y, z)
print("coefficients:", eansa)
center, axes, inve = polyToParams3D(eansa, False)  # get ellipsoid 3D parameters
print("center:", center)
print("axes:", axes)
print("rotationMatrix:", inve)


#%%
import plotly.graph_objects as go
import numpy as np
import plotly

fig = go.Figure(data=[go.Scatter3d(
    x=hull[0],
    y=hull[1],
    z=hull[2],
    mode='markers',
    marker=dict(
        size=12,
        # color=z,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename="/Users/yaolin/Downloads/plot.html", auto_open=True)


#%%

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        opacity=0.8
    )
)])

# from plotly.offline import iplot, init_notebook_mode
from plotly.graph_objs import Mesh3d

# some math: generate points on the surface of the ellipsoid
from math import pi
phi = np.linspace(0, 2*pi)
theta = np.linspace(-pi/2, pi/2)
phi, theta=np.meshgrid(phi, theta)

x = np.cos(theta) * np.sin(phi) * 32 * 4
y = np.cos(theta) * np.cos(phi) * 32 * 4
z = np.sin(theta) * .1767767 * 4

# to use with Jupyter notebook

fig.add_trace(Mesh3d({
                'x': x.flatten(),
                'y': y.flatten(),
                'z': z.flatten(),
                'alphahull': 0,
'opacity':.4}))

fig.update_layout(
            title={
                'text': "Simulation",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                # zaxis=dict(nticks=4, range=[-10, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1.2, y=1.2, z=.5),
        )

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename="/Users/yaolin/Downloads/test.html", auto_open=True)


















#%%
import numpy as np
from numpy.linalg import eig, inv
#least squares fit to an ellipse
# A x^2 + B xy + C y^2 + Dx + Ey = 1
#
# Returns coefficients A..F
# A x^2 + B xy + C y^2 + Dx + Ey + F = 0
# where F = -1

def ls_ellipse(xx,yy):

   # change xx from vector of length N to Nx1 matrix so we can use hstack
   x = xx[:,np.newaxis]
   y = yy[:,np.newaxis]

   J = np.hstack((x*x, x*y, y*y, x, y))
   K = np.ones_like(x) #column of ones

   JT=J.transpose()
   JTJ = np.dot(JT,J)
   InvJTJ=np.linalg.inv(JTJ);
   ABC= np.dot(InvJTJ, np.dot(JT,K))

   # ABC has polynomial coefficients A..E
   # Move the 1 to the other side and return A..F
   # A x^2 + B xy + C y^2 + Dx + Ey - 1 = 0
   eansa=np.append(ABC,-1)

   return eansa

def polyToParams(v,printMe):

   # convert the polynomial form of the ellipse to parameters
   # center, axes, and tilt
   # v is the vector whose elements are the polynomial
   # coefficients A..F
   # returns (center, axes, tilt degrees, rotation matrix)

   #Algebraic form: X.T * Amat * X --> polynomial form

   Amat = np.array(
   [
   [v[0],     v[1]/2.0, v[3]/2.0],
   [v[1]/2.0, v[2],     v[4]/2.0],
   [v[3]/2.0, v[4]/2.0, v[5]    ]
   ])

   if printMe: print('\nAlgebraic form of polynomial\n',Amat)

   #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
   A2=Amat[0:2,0:2]
   A2Inv=inv(A2)
   ofs=v[3:5]/2.0
   cc = -np.dot(A2Inv,ofs)
   if printMe: print('\nCenter at:',cc)

   # Center the ellipse at the origin
   Tofs=np.eye(3)
   Tofs[2,0:2]=cc
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))
   if printMe: print('\nAlgebraic form translated to center\n',R,'\n')

   R2=R[0:2,0:2]
   s1=-R[2, 2]
   RS=R2/s1
   (el,ec)=eig(RS)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)
   if printMe: print('\nAxes are\n',axes  ,'\n')

   rads=np.arctan2(ec[1,0],ec[0,0])
   deg=np.degrees(rads) #convert radians to degrees (r2d=180.0/np.pi)
   if printMe: print('Rotation is ',deg,'\n')

   inve=inv(ec) #inverse is actually the transpose here
   if printMe: print('\nRotation matrix\n',inve)
   return (cc[0],cc[1],axes[0],axes[1],deg,inve)

def printAns(pv,xin,yin,verbose):
  print('\nPolynomial coefficients, F term is -1:\n',pv)

  #normalize and make first term positive
  nrm=np.sqrt(np.dot(pv,pv))
  enrm=pv/nrm
  if enrm[0] < 0.0:
     enrm = - enrm
  print('\nNormalized Polynomial Coefficients:\n',enrm)

  #convert polynomial coefficients to parameterized ellipse (center, axes, and tilt)
  #also returns rotation matrix in last parameter
  # either pv or normalized parameter will work
  # params = polyToParams(enrm,verbose)

  params = polyToParams(pv,verbose)

  print("\nCenter at  %10.4f,%10.4f (truth is 1.5,  1.5)" % (params[0],params[1]))
  print("Axes gains %10.4f,%10.4f (truth is 1.55, 1.0)" % (params[2],params[3]))
  print("Tilt Degrees %10.4f (truth is 30.0)" % (params[4]))

  R=params[5]
  print('\nRotation Matrix\n',R)

  # Check solution
  # Convert to unit sphere centered at origin
  #  1) Subtract off center
  #  2) Rotate points so bulges are aligned with x, y axes (no xy term)
  #  3) Scale the points by the inverse of the axes gains
  #  4) Back rotate
  # Rotations and gains are collected into single transformation matrix M

  # subtract the offset so ellipse is centered at origin
  xc=xin-params[0]
  yc=yin-params[1]

  # create transformation matrix
  L = np.diag([1/params[2],1/params[3]])
  M=np.dot(R.T,np.dot(L,R))
  print('\nTransformation Matrix\n',M)

  # apply the transformation matrix
  [xm,ym]=np.dot(M,[xc,yc])
  # Calculate distance from origin for each point (ideal = 1.0)
  rm = np.sqrt(xm*xm + ym*ym)

  print("\nAverage Radius  %10.4f (truth is 1.0)" % (np.mean(rm)))
  print("Stdev of Radius %10.4f " % (np.std(rm)))
  return params


# if __name__ == '__main__':

# Test of least squares fit to an ellipse
# Samples have random noise added to both X and Y components
# True center is at (1.5, 1.5);
# X axis is 1.55, Y axis is 1.0, tilt is 30 degrees
# (or -150 from symmetry)
#
# Polynomial coefficients, F term is -1:
# A x^2 + B xy + C y^2 + Dx + Ey - 1 = 0
#
# A= -0.53968362, B=  0.50979868, C= -0.8285294
# D=  0.87914926, E=  1.72765849, F= -1

# Polynomial coefficients after normalization:
# A x^2 + B xy + C y^2 + Dx + Ey + F = 0
#
# A=  0.22041087, B= -0.20820563, C=  0.33837767
# D= -0.3590512,  E= -0.70558878  F=  0.40840756

# Test data, no noise
x0 = np.array(
[ 2.2255,   2.5995,   2.8634,   2.9163,
2.6252,   2.1366,   1.6252,   1.1421,
0.7084,   0.3479,   0.1094,   0.1072,
0.4497,   0.9500,   1.4583,   1.9341])

y0 = np.array(
[ 0.7817,   1.1319,   1.5717,   2.0812,
2.5027,   2.6578,   2.6150,   2.4418,
2.1675,   1.8020,   1.3480,   0.8351,
0.4534,   0.3381,   0.4061,   0.5973])

# Test data with added noise
xnoisy = np.array(
[ 2.2422,   2.5713,   2.8677,   2.9708,
2.7462,   2.2695,   1.7423,   1.2501,
0.8562,   0.4489,   0.0933,   0.0639,
0.3024,   0.7666,   1.2813,   1.7860])

ynoisy = np.array(
[ 0.7216,   1.1190,   1.5447,   2.0398,
2.4942,   2.7168,   2.6496,   2.5163,
2.1730,   1.8725,   1.5018,   0.9970,
0.5509,   0.3211,   0.3729,   0.5340])


print('\n==============================')
print('\nSolution for Perfect Data (to 4 decimal places)')

ans0= ls_ellipse(x0,y0)
printAns(ans0,x0,y0,0)

print('\n==============================')
print('\nSolution for Noisy Data')

ans = ls_ellipse(xnoisy,ynoisy)
p = printAns(ans,xnoisy,ynoisy,0)


#%%
plt.plot(xnoisy, ynoisy, 'r.-')
plt.plot(x0, y0, 'k.')

from matplotlib.patches import Ellipse
pl = Ellipse((p[0], p[1]), 2*p[2], 2*p[3], p[4])
plt.gca().add_patch(pl)
plt.show()
