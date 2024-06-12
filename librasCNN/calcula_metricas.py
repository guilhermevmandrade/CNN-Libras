def revocacao(TP, FN):
    """Sensibilidade, taxa de acerto, revocação, ou taxa de verdadeiros positivos (TPR)"""
    return TP / (TP + FN)

def especificidade(TN, FP):
    """Especificidade ou taxa de verdadeiros negativos (TNR)"""
    return TN / (TN + FP)

def precisao(TP, FP):
    """Precisão ou valor preditivo positivo (PPV)"""
    return TP / (TP + FP)

def acuracia(TP, TN, FP, FN):
    """Acurácia geral (ACC)"""
    return (TP + TN) / (TP + FP + FN + TN)

def f_measure(TP, FP, FN):
    """F-Measure ou F1-Score"""
    ppv = precisao(TP, FP)
    tpr = revocacao(TP, FN)
    return 2 * (ppv * tpr) / (ppv + tpr)
