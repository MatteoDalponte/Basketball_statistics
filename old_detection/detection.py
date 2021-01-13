import cv2
import numpy as np
import json

xc=0 #coordinata x del centro
yc=0 #coordinata y del centro
x_final = 0# coordinata punto alto a sx
y_final = 0
wc=0 #larghezza
hc=0 #altezza
score = 0

def detect(input, file_cfg, file_weights, file_classes):
    #FUNZIONE PER AVERE LE ETICHETTE IN OUTPUT
    def get_output_layers(net):

        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    #FUNZIONE PER DISEGNARE RETTANGOLO E SCRIVERE INFORMAZIONI A VIDEO
    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(classes[class_id])+" "+str(round(confidence,2)) #scrivo il nome della classe e relativa confidence
        fontColor = (0,0,255) #rosso
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), fontColor, 5)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, fontColor, 2)


    image = input

    Width = image.shape[1] #numero colonne
    Height = image.shape[0] #numero righe
    scale = 1/255 #fattore di scala

    classes = None

    with open(file_classes, 'r') as f: #vengono lette le classi ed inserite in un lista
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(file_weights, file_cfg) #creo deep neural network partendo dai file di configurazione e i pesi

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (1024, 1024)), scale , (1024,1024), (0,0,0), True, crop=False) #elaboro input

    net.setInput(blob) #definisco l'input alla rete

    outs = net.forward(get_output_layers(net)) #trovo le uscite passando il nome delle etichette in input

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.4 #originale -> 0.5
    nms_threshold = 0.4 #originale -> 0.4

    global xc
    global yc
    global x_final
    global y_final
    global wc
    global hc
    global score

    for out in outs:
        for detection in out:               # informazioni contenute nella singola detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    contatore=False
    ball_index_list=list()


    for i in indices:       # estraggo solo gli indici che hanno come id ball e limetto in h
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print(classes[class_ids[i]])
        if (classes[class_ids[i]])=="ball":
            ball_index_list.append(i)
            #try:
                #h.append(i)
                #print("append")
            #except AttributeError:
                #h=i
        

    for z in ball_index_list:
                    
        #riordino le detection in base al valore di confidence e prendo la più probabile        
        if (z==np.argmax(confidences)):
            box= boxes[z]
            x = box[0]      #angolo in alto a sx della palla
            y = box[1]      #angolo in alto a dx della palla
            w = box[2]
            h = box[3]
            wc=w         
            hc=h
            x_final = x
            y_final = y
            xc= int(x+(w/2))    # posizione x del centro delle palla
            yc= int(y+(h/2))    # posizione y del centro delle palla
            score = confidences[z]
            draw_prediction(image, class_ids[z], confidences[z], round(x), round(y), round(x+w), round(y+h))
            contatore=True
            print("posizione palla:"+str(round(x))+"   "+str(round(y))+"   "+str(round(x+w))+"   "+str(round(y+h)))
                    
    
    #return contatore, xc, yc, image, wc, hc, score     # return della posizione centralle della palla+scostamento w e h
    return contatore, x_final, y_final, image, wc, hc, score
