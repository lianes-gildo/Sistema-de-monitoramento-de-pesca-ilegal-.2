import pandas as pd, numpy as np, os, folium, math
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ---------------- Configurações ----------------
BASE="dados_pesca"; PERC=0.7; N_EST=200; CONT=0.2; RAND=42
np.random.seed(RAND)

# ---------------- Utilidades ----------------
def haversine(lat1, lon1, lat2, lon2):
    R=6371000; φ1,φ2=map(math.radians,[lat1,lat2])
    dφ,dλ=map(math.radians,[lat2-lat1,lon2-lon1])
    a=math.sin(dφ/2)**2+math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def ler_csvs():
    p=lambda f:os.path.join(BASE,f)
    return (pd.read_csv(p("embarcacoes_licencas.csv")),
            pd.read_csv(p("atividades_pesca.csv"),parse_dates=["data_atividade"]),
            pd.read_csv(p("rastreamento_ais.csv"),parse_dates=["timestamp"]),
            pd.read_csv(p("fiscalizacoes.csv"),parse_dates=["data_fiscalizacao"]))

# ---------------- Cálculo de features AIS ----------------
def calc_features(ais):
    feats=[]
    for m,g in ais.groupby("mmsi"):
        g=g.sort_values("timestamp")
        v=pd.to_numeric(g.velocidade_nos,errors="coerce").dropna()
        mean,std=(v.mean(),v.std()) if len(v) else (0,0)
        loiter=(v<1).mean() if len(v) else 0
        t=pd.to_datetime(g.timestamp,errors="coerce").dropna().values
        gaps=np.diff(t).astype("timedelta64[s]").astype(float)
        gap=np.clip(np.nanmean(gaps),0,36000) if len(gaps) else 0
        coords=list(zip(g.lat,g.lon))
        dist=sum(haversine(*coords[i],*coords[i+1]) for i in range(len(coords)-1))
        feats.append(dict(mmsi=m,velocidade_media_nos=mean,desvio_padrao_velocidade=std,
                          proporcao_loiter=loiter,dist_total_m=dist,avg_gap_s=gap))
    return pd.DataFrame(feats)

# ---------------- Integração ----------------
def integrar(lic,atv,aisf):
    df=(atv.merge(lic,on="id_embarcacao",how="left")
           .merge(aisf,left_on="id_embarcacao",right_on="mmsi",how="left")
           .drop(columns=["mmsi"],errors="ignore").fillna(0))
    df["hora_num"]=pd.to_datetime(df["hora_atividade"],format="%H:%M",errors="coerce").dt.hour
    return df

# ---------------- Heurística dinâmica e ponderada ----------------
def marcar_suspeitos(df):
    df=df.copy()
    df["score"]=np.where(df.licenca_valida.str.lower()=="não",0.45,0)
    df["score"]+=np.where(df.proporcao_loiter>0.4,0.15,0)
    df["score"]+=np.where(df.dist_total_m>8000,0.1,0)
    df["score"]+=np.where(df.avg_gap_s>900,0.15,0)
    df["score"]+=np.where(df.hora_num.isin(range(0,5))|(df.hora_num>=18),0.1,0)
    # refinamento: normalizar via z-score e misturar com percentil
    s=(df["score"]-df["score"].mean())/df["score"].std(ddof=0)
    th=max(df["score"].quantile(PERC)+(s.mean()/5),0.25)
    df["suspeito_heuristico"]=(df.score>=th).astype(int)
    df["status_doc"]=np.where(df.licenca_valida.str.lower()=="não","Irregular","Regular")
    df["status_comp"]=np.where(df.suspeito_heuristico==1,"Suspeito","Normal")
    df["categoria"]=np.select(
        [(df.status_doc=="Irregular")&(df.status_comp=="Suspeito"),
         (df.status_doc=="Irregular"),(df.status_comp=="Suspeito")],
        ["Irregular+Suspeito","IrregularDocumental","SuspeitoComportamental"],"Regular")
    return df

# ---------------- Modelo RandomForest ----------------
def treino(df):
    feats=["velocidade_media_nos","desvio_padrao_velocidade","proporcao_loiter","dist_total_m","avg_gap_s"]
    X=df[feats].fillna(0); y=df.ocorrencia_infracao.astype(str).str.lower().map({"sim":1,"não":0,"nao":0}).fillna(0).astype(int)
    if y.nunique()<2: df["pred_rf"]=0; print("Dados insuficientes."); return df
    X=StandardScaler().fit_transform(X)
    rf=RandomForestClassifier(n_estimators=N_EST,class_weight="balanced",random_state=RAND)
    skf=StratifiedKFold(5,shuffle=True,random_state=RAND)
    cv=cross_val_score(rf,X,y,cv=skf,scoring="accuracy").mean()
    rf.fit(X,y); pred=rf.predict(X)
    df["pred_rf"]=pred
    print("\n--- RandomForest ---\n",classification_report(y,pred,zero_division=0))
    print("Cross-val média:",round(cv,3))
    return df

# ---------------- Detecção de Anomalias ----------------
def anomalias(df):
    X=df[["velocidade_media_nos","desvio_padrao_velocidade","proporcao_loiter","dist_total_m","avg_gap_s"]].fillna(0)
    iso=IsolationForest(contamination=CONT,random_state=RAND)
    iso.fit(X); df["anomalia_score"]=-iso.decision_function(X)
    print("Média score de anomalia:",round(df.anomalia_score.mean(),3))
    return df

# ---------------- Validação ----------------
def validar(df,fis):
    val=df.merge(fis[["id_embarcacao","data_fiscalizacao","resultado"]],
                 left_on=["id_embarcacao","data_atividade"],
                 right_on=["id_embarcacao","data_fiscalizacao"],how="left")
    ac=(val.suspeito_heuristico.eq(1)&val.resultado.eq("Infracao")).sum()
    pct=round(ac/len(val)*100,2)
    print(f"Heurística acertou {ac}/{len(val)} fiscalizações ({pct}%)")
    return val

# ---------------- Geração de mapa ----------------
def mapa(df, out="mapa_barcos.html"):
    m = folium.Map(location=[-17.5, 36], zoom_start=6)

    # cores por categoria
    cores = {
        "Irregular+Suspeito": "red",
        "SuspeitoComportamental": "orange",
        "IrregularDocumental": "gray",
        "Regular": "blue"
    }

    # gerar marcadores
    for _, r in df.groupby("id_embarcacao").first().iterrows():
        cor = cores.get(r.categoria, "blue")
        tip = (
            f"<b>{r.nome_embarcacao}</b><br>"
            f"Tipo de embarcação: {r.tipo_embarcacao}<br>"
            f"Licença: {r.licenca_valida}<br>"
            f"Categoria de status: {r.categoria}<br>"
            f"Velocidade média: {r.velocidade_media_nos:.1f} nós<br>"
            f"Proporção de loiter: {r.proporcao_loiter:.2f}<br>"
            f"Distância total percorrida: {r.dist_total_m:.0f} m<br>"
            f"Score de anomalia: {r.anomalia_score:.2f}<br>"
            f"Score heurístico: {r.score:.2f}"
        )
        folium.CircleMarker(
            [r.coordenada_lat, r.coordenada_lon],
            radius=7,
            color=cor,
            fill=True,
            fill_color=cor,
            fill_opacity=0.7,
            tooltip=tip
        ).add_to(m)

    # adicionar legenda no canto inferior esquerdo
    legenda_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 130px; 
                border:2px solid grey; z-index:9999; font-size:14px; 
                background-color:white; padding:10px;">
      <b>Legenda do Status</b><br>
      <i style="background:red;color:red;">&nbsp;&nbsp;&nbsp;</i>&nbsp; Irregular + Suspeito<br>
      <i style="background:orange;color:orange;">&nbsp;&nbsp;&nbsp;</i>&nbsp; Suspeito Comportamental<br>
      <i style="background:gray;color:gray;">&nbsp;&nbsp;&nbsp;</i>&nbsp; Irregular Documental<br>
      <i style="background:blue;color:blue;">&nbsp;&nbsp;&nbsp;</i>&nbsp; Regular
    </div>
    """
    m.get_root().html.add_child(folium.Element(legenda_html))

    m.save(out)

# ---------------- Pipeline Principal ----------------
def main():
    lic,atv,ais,fis=ler_csvs()
    df=integrar(lic,atv,calc_features(ais))
    df=marcar_suspeitos(df)
    df=anomalias(treino(df))
    df=validar(df,fis)
    os.makedirs(BASE,exist_ok=True)
    df.to_csv(os.path.join(BASE,"integracao_features.csv"),index=False)
    mapa(df,os.path.join(BASE,"mapa_barcos.html"))

if __name__=="__main__": main()
