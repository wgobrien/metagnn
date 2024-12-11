import pyro
import pyro.poutine as poutine

def get_learned_params(vae, batch):
    vae.encoder.eval()
    vae.decoder.eval()
    guide_trace = poutine.trace(vae.guide).get_trace(batch["graphs"], batch["graph_sim"])
    trained_model = poutine.replay(vae.model, trace=guide_trace)
    trained_trace = poutine.trace(trained_model).get_trace(batch["graphs"], batch["graph_sim"])
    params = trained_trace.nodes
    return params