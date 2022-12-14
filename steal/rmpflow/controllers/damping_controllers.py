from steal.rmpflow.controllers.force_controllers import *
from steal.rmpflow.controllers.metrics import *
from steal.rmpflow.controllers.momentum_controllers import *
from steal.rmpflow.controllers.potentials import *


class DampingForceController(NaturalGradientDescentForceController):

    def __init__(self,
                 damping_gain=0.0,
                 ds_type='gds',
                 device=torch.device('cpu')):
        self.damping_gain = damping_gain
        # The Riemannian metric
        G = IdentityMetric(scaling=1., device=device)
        Phi = ZeroPotential(device=device)
        del_Phi = Phi.grad

        def B(x, xd):
            return self.damping_gain * G(x, xd)

        super(DampingForceController, self).__init__(G=G,
                                                     B=B,
                                                     del_Phi=del_Phi,
                                                     ds_type=ds_type,
                                                     device=device)


# Techinically not damping
class DampingMomemtumController(NaturalGradientDescentMomentumController):

    def __init__(self, damping_gain=0.0, device=torch.device('cpu')):
        self.damping_gain = damping_gain
        G = IdentityMetric(scaling=self.damping_gain, device=device)
        Phi = ZeroPotential(device=device)
        del_Phi = Phi.grad
        super(DampingMomemtumController, self).__init__(G=G,
                                                        del_Phi=del_Phi,
                                                        device=device)
