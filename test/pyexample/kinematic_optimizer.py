import carticulate as cart
import numpy as np
from scipy.spatial.transform import Rotation as R

pose_gt = np.array([
    0.406466513872, -0.852810859680,  0.146360948682,  0.293388634920,
    0.227000191808, -0.091430991888, -0.126407116652, -0.961318075657,
    0.559360682964, -0.335615158081, -0.593093216419, -0.471930682659,
    0.520075201988, -0.525091350079,  0.615168750286,  0.274532765150,
    0.224615797400, -0.316543459892, -0.916824698448, -0.093704506755,
    0.730132281780, -0.406292289495, -0.539213001728,  0.105274818838,
    0.317469179630, -0.555329620838, -0.548433601856, -0.538556337357,
    0.880365848541,  0.048340465873, -0.085648462176,  0.463986545801,
    0.573147833347, -0.461526453495, -0.341062277555, -0.584954261780,
    0.627840995789,  0.718061864376, -0.075795009732, -0.290616571903,
    0.422042757273, -0.721613466740, -0.499400019646,  0.227494016290,
    0.358688920736, -0.041317056865, -0.261775255203, -0.895046889782,
    0.716889619827,  0.461497515440,  0.076190352440, -0.516995429993,
    0.023606089875, -0.517991423607,  0.271160930395,  0.810925006866,
    0.261237382889, -0.328141123056,  0.485250055790,  0.767209708691,
    0.874739468098,  0.451379001141,  0.044348072261, -0.170649155974,
    0.193883493543, -0.937609195709,  0.288603872061, -0.002450699219,
    0.343840390444, -0.663194835186, -0.649738967419, -0.140662044287,
    0.554252982140, -0.274869352579, -0.718913674355, -0.316881120205,
    0.201899036765, -0.273315548897, -0.397826313972, -0.852214515209,
    0.742405295372,  0.472843199968, -0.467474848032, -0.081981733441,
    0.444336771965, -0.544762492180, -0.625518560410,  0.338415622711,
    0.035868179053, -0.743403136730,  0.667864143848,  0.004765785299,
    0.587408542633,  0.357875138521,  0.599712193012,  0.408927708864
]).reshape(24, 4)
tran_gt = np.array([-1, 2, 3.])

model = cart.KinematicModel(r'C:\Users\Admin\Work\projects\carticulate\armatures\SMPL_male.armature')
model.set_state_q(pose_gt, tran_gt)
optimizer = cart.KinematicOptimizer(model, verbose=True)
for i in range(24):
    if True:   # orientation observation
        q = np.random.randn(4)
        local_orientation = q / np.linalg.norm(q)
        observation = model.get_orientation_q(i, local_orientation)
        optimizer.add_observation(cart.OrientationObservation(i, local_orientation, observation, weight=10))

    if True:   # position 3D observation
        for j in range(3):
            local_position = np.random.randn(3)
            observation = model.get_position(i, local_position)
            optimizer.add_observation(cart.Position3DObservation(i, local_position, observation, weight=10))

    if True:   # position 2D observation
        KT = np.array([[10, 0, 10, -10],
                       [0, 10, 10, -10],
                       [0,  0,  1, -10.]])
        for j in range(5):
            local_position = np.random.randn(3)
            p = KT[:, :3] @ model.get_position(i, local_position) + KT[:, 3]
            uv = p[:2] / p[2:]
            optimizer.add_observation(cart.Position2DObservation(i, local_position, uv, KT, cart.RobustKernel.HUBER, 1, weight=1))

optimizer.print()
model.set_state_q(np.broadcast_to(np.array([1, 0, 0, 0.]), (24, 4)), np.zeros(3))
optimizer.set_constraints([True] * 24, True)
optimizer.optimize(50)

pose_pred, tran_pred = model.get_state_q()
for i in range(24):
    dR = R.from_quat(pose_gt[i][[1, 2, 3, 0]]) * R.from_quat(pose_pred[i][[1, 2, 3, 0]]).inv()
    print('joint %d err:' % i, dR.as_rotvec())
print('tran err:', tran_gt - tran_pred)
