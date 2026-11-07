"""Mamdani próprio / referência scikit-fuzzy: 5 regras para priorização ANALÍTICA."""
import numpy as np

def tri(x,a,b,c):
    x=np.asarray(x,float)
    left=np.where(b>a,(x-a)/max(b-a,1e-9),np.where(x>=b,1.,0.))
    right=np.where(c>b,(c-x)/max(c-b,1e-9),np.where(x<=b,1.,0.))
    return np.clip(np.minimum(left,right),0,1)

def activation(risk,trend):
    r={'baixo':tri(risk,0,0,.5).item(),'medio':tri(risk,.2,.5,.8).item(),'alto':tri(risk,.5,1,1).item()}
    t={'estavel':tri(trend,0,0,.65).item(),'crescente':tri(trend,.35,1,1).item()}
    # Cinco regras: a segunda é uma disjunção de duas condições equivalentes de saída.
    return {
      'baixa':min(r['baixo'],t['estavel']),
      'media':max(min(r['baixo'],t['crescente']),min(r['medio'],t['estavel']),min(r['alto'],t['estavel'])),
      'alta':max(min(r['medio'],t['crescente']),min(r['alto'],t['crescente']))
    }

def mamdani(risk,trend):
    degrees=activation(float(risk),float(trend));x=np.linspace(0,100,1001)
    out=np.maximum.reduce([
      np.minimum(degrees['baixa'],tri(x,0,0,45)),
      np.minimum(degrees['media'],tri(x,20,50,80)),
      np.minimum(degrees['alta'],tri(x,55,100,100))])
    den=float(np.trapezoid(out,x))
    return float(np.trapezoid(out*x,x)/den) if den>1e-10 else 50.

def mamdani_skfuzzy(risk,trend):
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    x=np.linspace(0,1,1001); u=np.linspace(0,100,1001)
    r=ctrl.Antecedent(x,'risco');t=ctrl.Antecedent(x,'tendencia');s=ctrl.Consequent(u,'prioridade')
    for key,v in {'baixo':(0,0,.5),'medio':(.2,.5,.8),'alto':(.5,1,1)}.items():r[key]=fuzz.trimf(x,v)
    for key,v in {'estavel':(0,0,.65),'crescente':(.35,1,1)}.items():t[key]=fuzz.trimf(x,v)
    for key,v in {'baixa':(0,0,45),'media':(20,50,80),'alta':(55,100,100)}.items():s[key]=fuzz.trimf(u,v)
    rules=[ctrl.Rule(r['baixo']&t['estavel'],s['baixa']),
           ctrl.Rule((r['baixo']&t['crescente'])|(r['medio']&t['estavel']),s['media']),
           ctrl.Rule(r['medio']&t['crescente'],s['alta']),
           ctrl.Rule(r['alto']&t['crescente'],s['alta']),
           ctrl.Rule(r['alto']&t['estavel'],s['media'])]
    sim=ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    sim.input['risco']=float(risk);sim.input['tendencia']=float(trend);sim.compute()
    return float(sim.output['prioridade'])
