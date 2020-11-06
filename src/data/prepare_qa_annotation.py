"""
Create the data necessary for the annotation of service-public.fr
Directly modify in the main the parameters for the path of the information you are looking for. Parameters :
    path_arbo = Path('./data/arborescence_spf_particuliers.json')
    path_dataset = Path('./data/407_question-fiche_anonym.csv')
    path_knowledge = Path('./data/v10')

the output data is a json with the following information :

{
  "version": "v1.0",
  "data": [{
    'displaytitle': 'Transports',
    'sous_theme': '',
    'categorie': "Carte grise (certificat d'immatriculation)",
    'sous_dossier': 'Immatriculer un véhicule',
    'reference': "Immatriculer un véhicule d'occasion",
    'id': 'F1050',
    'paragraphs': [{
        'context': "Carte grise\xa0: immatriculer un véhicule d'occasion : \n\n\nSi vous achetez une voiture en France ou à l'étranger, vous avez 1 mois pour la faire immatriculer et obtenir ainsi une carte grise (certificat d'immatriculation). En cas de contrôle routier, l'absence de présentation de la carte grise peut entraîner une amende pouvant aller jusqu'à  750\xa0€ . Vous risquez aussi l'immobilisation immédiate du véhicule. La démarche ne s'effectue plus en préfecture mais auprès de l'ANTS, par voie dématérialisée.\nEn ligne\nUn dispositif de copie numérique (scanner, appareil photo numérique, smartphone ou tablette avec fonction photo) est nécessaire. Le format des documents numérisés à transmettre peut être un des suivants\xa0: JPG, PNG, BMP, TIFF, PDF.\nDes  points numériques \n\n (avec ordinateurs, imprimantes et scanners) sont mis à votre disposition dans chaque préfecture et dans la plupart des sous-préfectures.Vous pouvez y accomplir la démarche, aidé par des médiateurs si vous rencontrez des difficultés avec l'utilisation d'internet.\nVous pouvez aussi être accompagné dans votre démarche dans une maison de services au public.\nVous devez faire la démarche sur le site de l'ANTS en vous identifiant via  FranceConnect .\nVous devez disposer d'une copie numérique (photo ou scan) de plusieurs documents  :\nFormulaire  cerfa n°13750 \n\n\n Justificatif de domicile  de moins de 6 mois (ou, en cas de cotitulaires, justificatif de celui dont l'adresse va figurer sur la carte grise)\nCarte grise  du véhicule, barrée avec la mention  Vendu le (jour/mois/année)   ou  Cédé le (jour/mois/année) , et avec la signature de l'ancien propriétaire  (ou de tous les cotitulaires s'il y en avait)\nSi le véhicule a plus de 4 ans, preuve du contrôle technique en cours de validité, sauf si le véhicule en est dispensé . Le contrôle technique doit dater de moins de 6 mois le jour de la demande de carte grise\xa0: si le délai est dépassé, il faudra réaliser un nouveau contrôle à vos frais\nSi vous faites la démarche pour quelqu'un d'autre,  mandat  signé et  pièce d'identité  de la personne pour qui vous effectuez la démarche\nVous devez disposer du  code de cession , remis par l'ancien propriétaire du véhicule.\nVous devez certifier sur l'honneur que le demandeur de la carte grise dispose\nd'une attestation d'assurance du véhicule\net d'un permis de conduire correspondant à la catégorie du véhicule immatriculé.\nVous n'avez pas à joindre une copie numérique (photo ou scan) du permis de conduire. En revanche, celle-ci pourra vous être demandée lors de l'instruction de votre dossier.\nVous devez par ailleurs fournir des informations, notamment\xa0:\nl'identité du titulaire (et éventuellement des co-titulaires) de la carte grise\xa0: nom, prénoms, sexe, date et lieu  de naissance, numéro de téléphone et adresse électronique,\nvos coordonnées bancaires.\nLe règlement du  montant de la carte grise  doit obligatoirement être effectué par carte bancaire.\nLe coût de la carte grise est variable. Il dépend notamment des caractéristiques du véhicule et de la région dans laquelle vous vivez.\nVous pouvez évaluer le coût de votre carte grise en utilisant ce simulateur\xa0:\nÀ la fin de la procédure, vous obtenez\xa0:\nun numéro de dossier,\nun accusé d'enregistrement de votre demande\net un certificat provisoire d'immatriculation (CPI), que vous devez  imprimer. Le CPI vous permet de circuler pendant 1 mois, uniquement en France, en attendant de recevoir votre carte grise.\nVous recevrez votre carte grise sous  pli sécurisé   en général dans les  7  jours ouvrés . Toutefois, le délai peut être plus long\nsi votre demande est incomplète ou doit être analysée par le service instructeur (la fabrication de la carte grise ne pourra être lancée qu'à l'issue de cette étape)\nou en fonction du nombre de demandes en cours de traitement.\nVous pouvez suivre l'état d'avancement de votre dossier sur le site de l'Agence nationale des titres sécurisés (ANTS)\xa0: \nMunissez-vous du certificat provisoire d'immatriculation (CPI).\nSi vous êtes absent lors du passage du facteur, un avis de passage vous sera déposé. Vous avez ensuite 15 jours pour récupérer votre document à La Poste (ou donner procuration à un tiers pour le faire à votre place). Passé ce délai, le titre est retourné à l'expéditeur. Vous devrez contacter l'ANTS pour qu'il vous soit renvoyé.\nPar messagerie\n\n\t\t\t\n      Accès au\n       formulaire en ligne \n\n\t\t\nvous devez conserver l'ancienne carte grise pendant 5 ans, puis la détruire.\nAuprès d'un professionnel habilité\nEn plus du coût de la carte grise, le professionnel vous facturera une somme correspondant à la prestation qu'il réalise à votre place. Cette somme est librement fixée par le professionnel.\nVous devez présenter les documents suivants\xa0: \nCarte grise  du véhicule, barrée avec la mention  Vendu le (jour/mois/année)   ou  Cédé le (jour/mois/année)  ,  avec  la signature de l'ancien propriétaire (ou de tous les co-titulaires s'il y en avait)\nFormulaire  cerfa 13757  de mandat à un professionnel\n\n Justificatif d'identité   (original), un par cotitulaire\n\n Justificatif de domicile  (original ), ou en cas de co-titulaires, le justificatif de celui dont l'adresse va figurer sur la carte grise\n\n Coût de la carte grise , en chèque ou par carte bancaire\n\n Preuve du contrôle technique , si le véhicule a plus de 4 ans et  n'en est pas dispensé . Le contrôle doit avoir moins de 6 mois (2 mois si une  contre-visite  a été prescrite) et doit avoir été réalisé en France. Il doit dater de moins de 6 mois le jour de la demande de carte grise\xa0: si le délai est dépassé, il faudra réaliser un nouveau contrôle à vos frais.\nFormulaire  cerfa n°13750 \n\nFormulaire  cerfa n°15776  de déclaration de cession du véhicule rempli et signé par l'ancien et le nouveau propriétaire (s'il y a plusieurs copropriétaires, chacun doit le signer). Si vous avez perdu le formulaire rempli et signé, vous devrez vous contacter le vendeur pour refaire le formulaire.\nAttestation d'assurance du véhicule à immatriculer\nPermis de conduire correspondant à la catégorie du véhicule à immatriculer\nVous devez par ailleurs fournir des informations, notamment\xa0l'identité du titulaire (et éventuellement des co-titulaires) de la carte grise\xa0: nom, prénoms, sexe, date et lieu  de naissance, numéro de téléphone et adresse électronique.\nUn  certificat provisoire d'immatriculation (CPI)  sera mis à disposition immédiatement.  Il vous permet de circuler pendant un mois (en France uniquement) en attendant de recevoir votre carte grise définitive.\nVous recevrez votre carte grise définitive sous  pli sécurisé  à votre domicile sous  un délai qui peut varier . Vous pouvez  suivre en ligne l'état d'avancement de sa réalisation .\nvous devez conserver l'ancienne carte grise pendant 5 ans, puis la détruire.\nComment faire la démarche\xa0?\nIl n'est désormais plus possible de demander une carte grise auprès de la préfecture ou de la sous-préfecture.\nVous devez faire la démarche sur le site de l'ANTS en vous identifiant via  FranceConnect .\nUn dispositif de copie numérique (scanner, appareil photo numérique, smartphone ou tablette avec fonction photo) est nécessaire. Le format des documents numérisés à transmettre peut être un des suivants\xa0: JPG, PNG, BMP, TIFF, PDF.\nDes  points numériques \n\n (avec ordinateurs, imprimantes et scanners) sont mis à votre disposition dans chaque préfecture et dans la plupart des sous-préfectures.Vous pouvez y accomplir la démarche, aidé par des médiateurs si vous rencontrez des difficultés avec l'utilisation d'internet.\nVous pouvez aussi être accompagné dans votre démarche dans une maison de services au public.\nDans quel délai\xa0?\nVous avez 1 mois pour faire la démarche. Si vous ne faites pas la démarche à temps et que vous êtes contrôlé par les forces de l'ordre,  vous risquez une amende pouvant  aller jusqu'à  750\xa0€  (en général,   amende forfaitaire  de  135\xa0€ ).\nPièces à fournir et obtention du titre\nLa liste des pièces diffère selon que le véhicule était précédemment immatriculé dans un pays de  l'Union européenne  ou dans un autre pays.\nVéhicule immatriculé dans un pays de l'Union européenne\nVous devez vous munir d'une copie numérique (photo ou scan) des documents suivants\xa0: \n Carte grise d'origine, sans mention particulière, ou une pièce officielle de propriété du véhicule\nSi la carte grise a été conservée par les autorités administratives du pays étranger,  document officiel l'indiquant ou certificat international pour automobiles en cours de validité délivré par ces autorités\n\n Justificatif de domicile  de moins de 6 mois (ou, en cas de cotitulaires, justificatif de celui dont l'adresse va figurer sur la carte grise)\nFormulaire  cerfa n°13750 \n\n\n Preuve du contrôle technique , si le véhicule a plus de 4 ans et  n'en est pas dispensé . Le contrôle doit avoir moins de 6 mois (quand une  contre-visite  a été prescrite, le délai accordé pour l'effectuer ne doit pas être dépassé) et doit avoir été réalisé en France ou dans l'Union européenne si le véhicule y était immatriculé. Le contrôle technique doit dater de moins de 6 mois le jour de la demande de carte grise\xa0: si le délai est dépassé, il faudra réaliser un nouveau contrôle à vos frais.\nSauf si la demande d'immatriculation est revêtue d'une mention de dispense attribuée par les services fiscaux,  quitus fiscal  délivré par la recette principale des impôts attestant que la TVA a bien été payée en France. Le quitus n'est pas à fournir pour une remorque ou semi-remorque.\nSi vous faites la démarche pour quelqu'un d'autre,  mandat  signé et  pièce d'identité  de la personne pour qui vous effectuez la démarche\nSi l'ancienne carte grise ne peut pas être fournie ou ne correspond pas au véhicule importé ou ne permet pas de l'identifier ou ne comporte pas toutes les données obligatoires, justificatif complémentaire correspondant à votre situation :\nCertificat de conformité européen délivré par le constructeur, édité si nécessaire dans une autre langue que le français. Il peut être délivré sous forme de document numérique.\nAttestation d'identification à un type communautaire\nProcès-verbal de réception à titre isolé (RTI) établi par une  Dreal .\nSi vous habitez en  Île-de-France , vous devez demander le procès-verbal de réception à titre isolé -RTI -à la plate-forme régionale de réception de véhicules de la  DRIEE Ile de France .\nVous devez certifier sur l'honneur que le demandeur de la carte grise dispose\nd'une attestation d'assurance du véhicule\net d'un permis de conduire correspondant à la catégorie du véhicule immatriculé.\nVous n'avez pas à joindre une copie numérique (photo ou scan) du permis de conduire. En revanche, celle-ci pourra vous être demandée lors de l'instruction de votre dossier.\nVous devez par ailleurs fournir des informations, notamment\xa0:\nl'identité du titulaire (et éventuellement des cotitulaires) de la carte grise\xa0: nom, prénoms, sexe, date et lieu  de naissance, numéro de téléphone et adresse électronique,\nvos coordonnées bancaires.\nLe règlement du  montant de la carte grise  doit obligatoirement être effectué par carte bancaire.\nLe coût de la carte grise est variable. Il dépend notamment des caractéristiques du véhicule et de la région dans laquelle vous vivez.\nVous pouvez évaluer le coût de votre carte grise en utilisant ce simulateur\xa0:\nÀ la fin de la procédure, vous obtenez\xa0:\nun numéro de dossier,\nun accusé d'enregistrement de votre demande\net un certificat provisoire d'immatriculation (CPI), que vous devez  imprimer. Le CPI vous permet de circuler pendant 1 mois, uniquement en France, en attendant de recevoir votre carte grise.\nVous recevrez votre carte grise sous  pli sécurisé   en général dans les  7  jours ouvrés . Toutefois, le délai peut être plus long\nsi votre demande est incomplète ou doit être analysée par le service instructeur (la fabrication de la carte grise ne pourra être lancée qu'à l'issue de cette étape)\nou en fonction du nombre de demandes en cours de traitement.\nVous pouvez suivre l'état d'avancement de votre dossier sur le site de l'Agence nationale des titres sécurisés (ANTS)\xa0: \nMunissez-vous du certificat provisoire d'immatriculation (CPI).\nSi vous êtes absent lors du passage du facteur, un avis de passage vous sera déposé. Vous avez ensuite 15 jours pour récupérer votre document à La Poste (ou donner procuration à un tiers pour le faire à votre place). Passé ce délai, le titre est retourné à l'expéditeur. Vous devrez contacter l'ANTS pour qu'il vous soit renvoyé.\nDans un autre pays\nVous devez vous munir d'une copie numérique (photo ou scan) des documents suivants\xa0:\n Carte grise d'origine, sans mention particulière, ou pièce officielle de propriété du véhicule\nSi la carte grise a été conservée par les autorités administratives du pays étranger,  document officiel l'indiquant, ou certificat international pour automobiles en cours de validité délivré par ces autorités\n\n Justificatif de domicile  de moins de 6 mois (ou, en cas de cotitulaires, justificatif de celui dont l'adresse va figurer sur la carte grise)\nFormulaire  cerfa n°13750 \n\n\n Preuve du contrôle technique , si le véhicule a plus de 4 ans et  n'en est pas dispensé . Le contrôle doit avoir moins de 6 mois (quand une  contre-visite  a été prescrite, le délai accordé pour l'effectuer ne doit pas être dépassé) et doit avoir été réalisé en France ou dans un pays de l'Union européenne. Le contrôle technique doit dater de moins de 6 mois le jour de la demande de carte grise\xa0: si le délai est dépassé, il faudra réaliser un nouveau contrôle à vos frais.\nCertificat de dédouanement 846 A, sauf si la demande d'immatriculation est revêtue d'une mention de dispense attribuée par les services des douanes\nSi vous faites la démarche pour quelqu'un d'autre,  mandat  signé et  pièce d'identité  de la personne pour qui vous effectuez la démarche\nJustificatif technique de conformité correspondant à la situation du véhicule :\ncertificat de conformité européen, délivré par le constructeur\nou attestation d'identification du véhicule au type communautaire, délivrée par le constructeur ou son représentant en France ou une  Dreal \n\nou procès-verbal de réception à titre isolé (RTI) délivré par une  Dreal \n\nSi vous habitez en  Île-de-France , vous devez demander le procès-verbal de RTI à la plate-forme régionale de réception de véhicules de la  DRIEE Ile de France .\nVous devez certifier sur l'honneur que le demandeur de la carte grise dispose\nd'une attestation d'assurance du véhicule\net d'un permis de conduire correspondant à la catégorie du véhicule immatriculé.\nVous n'avez pas à joindre une copie numérique (photo ou scan) du permis de conduire. En revanche, celle-ci pourra vous être demandée lors de l'instruction de votre dossier.\nVous devez par ailleurs fournir des informations, notamment:\nl'identité du titulaire (et éventuellement des co-titulaires) de la carte grise\xa0: nom, prénoms, sexe, date et lieu  de naissance, numéro de téléphone et adresse électronique,\nvos coordonnées bancaires.\nLe règlement du  montant de la carte grise  doit obligatoirement être effectué par carte bancaire.\nLe coût de la carte grise est variable. Il dépend notamment des caractéristiques du véhicule et de la région dans laquelle vous vivez.\nVous pouvez évaluer le coût de votre carte grise en utilisant ce simulateur\xa0:\nÀ la fin de la procédure, vous obtenez\xa0:\nun numéro de dossier,\nun accusé d'enregistrement de votre demande\net un certificat provisoire d'immatriculation (CPI), que vous devez  imprimer. Le CPI vous permet de circuler pendant 1 mois, uniquement en France, en attendant de recevoir votre carte grise.\nVous recevrez votre carte grise sous  pli sécurisé   en général dans les  7  jours ouvrés . Toutefois, le délai peut être plus long\nsi votre demande est incomplète ou doit être analysée par le service instructeur (la fabrication de la carte grise ne pourra être lancée qu'à l'issue de cette étape)\nou en fonction du nombre de demandes en cours de traitement.\nVous pouvez suivre l'état d'avancement de votre dossier sur le site de l'Agence nationale des titres sécurisés (ANTS)\xa0: \nMunissez-vous du certificat provisoire d'immatriculation (CPI).\nSi vous êtes absent lors du passage du facteur, un avis de passage vous sera déposé. Vous avez ensuite 15 jours pour récupérer votre document à La Poste (ou donner procuration à un tiers pour le faire à votre place). Passé ce délai, le titre est retourné à l'expéditeur. Vous devrez contacter l'ANTS pour qu'il vous soit renvoyé.\nCoût\nLe coût de la carte grise est variable. Il dépend notamment des caractéristiques du véhicule et de la région dans laquelle vous vivez.\nVous pouvez évaluer le coût de votre carte grise en utilisant ce simulateur\xa0:\nServeur vocal interactif national apportant des réponses automatisées concernant la carte grise, le permis de conduire, la carte nationale d'identité et le passeport. \nSi l'usager n'obtient pas la réponse à sa question relative à la carte grise ou au permis de conduire, il sera mis en relation avec un conseiller de l'Agence nationale des titres sécurisés (ANTS).\nPar téléphone\nDepuis la métropole\xa0:\n\n 34 00  (coût d'un appel local)\nDepuis l'outre-mer\xa0:\n\n 09 70 83 07 07 \n\nDepuis l'étranger\xa0:\n\n +33 9 70 83 07 07 \n\nAllemagne, Autriche, Belgique, Bulgarie, Chypre, Croatie, Danemark, Espagne, Estonie, Finlande, France, Grèce, Hongrie, Irlande, Italie, Lituanie, Lettonie, Luxembourg, Malte, Pays-Bas, Pologne, Portugal, République tchèque, Roumanie,  Slovaquie, Slovénie, Suède. Attention\xa0: le Royaume-uni a quitté  l'Union européenne, mais le droit européen concernant les citoyens  s'applique jusqu'au 31 décembre 2020.\nJour effectivement travaillé dans une entreprise ou une administration.\n\t\tOn en compte 5 par semaine.\nSecond examen nécessaire pour vérifier si les points défectueux détectés (défaillance majeure ou critique) lors de la visite initiale du véhicule ont été réparés\nCourrier suivi remis contre signature\nCode informatique obtenu par le vendeur particulier à l'issue de la démarche de la téléprocédure de la vente d'un véhicule d'occasion. Il doit être communiqué au nouveau propriétaire du véhicule afin qu'il réalise sa demande de carte grise.\nConnexion avec l'identifiant et le mot de passe de votre compte Impots.gouv.fr ou Ameli.fr ou Iidentitenumerique.laposte.fr ou Mobileconnectetmoi.fr ou Msa.fr ou Alicem\nDépartements 75, 77, 78, 91, 92, 93, 94 et 95\nDirection régionale de l'environnement, de l'aménagement et du logement\nDirection régionale et interdépartementale de l'environnement et de l'énergie",
        'qas': [
            "J'ai acheter un véhicule mais j'ai égarer le certificat de cession je dois faire la carte grise a mon nom et je n'arrive pas a joindre le vendeur comment dois je faire?"
            ]
        }]
    }]
}"""


import pandas as pd
import json
from pathlib import Path


def get_arbo(arbo):
    res = {'theme': '', 'sous_theme': '', 'dossier': '', 'sous_dossier': '', 'fiche': '', 'id': ''}
    for level_dict in arbo:
        res[level_dict['type']] = level_dict['name']
        if level_dict['type'] == 'fiche':
            res['id'] = level_dict['id']
    # rename keys to match annotation scheme
    res['displaytitle'] = res.pop('theme')
    res['categorie'] = res.pop('dossier')
    res['reference'] = res.pop('fiche')
    return res


def get_arborescence(arborescence, fiche_id):
    arborescence = arborescence['data']
    for level_1_dict in arborescence:
        arbo = [level_1_dict] #used to remember the path to the fiche
        for level_2_dict in level_1_dict['data']:
            if len(arbo) > 1:
                arbo= arbo[:1] #keep only the first level
            arbo.append(level_2_dict)
            for level_3_dict in level_2_dict['data']:
                if len(arbo) > 2:
                    arbo= arbo[:2] #keep only up to the 2nd level
                arbo.append(level_3_dict)
                if level_3_dict['id'] == fiche_id:
                    return get_arbo(arbo)
                elif level_3_dict['type'] == 'fiche':
                    continue
                else:
                    for level_4_dict in level_3_dict['data']:
                        if len(arbo) > 3:
                            arbo= arbo[:3] #keep only up to the 3rd level
                        arbo.append(level_4_dict)
                        if level_4_dict['id'] == fiche_id:
                            return get_arbo(arbo)
                        elif level_4_dict['type'] == 'fiche':
                            continue
                        else:
                            try:
                                for level_5_dict in level_4_dict['data']:
                                    if len(arbo) > 4:
                                        arbo= arbo[:4] #keep only up to the 4th level
                                    arbo.append(level_5_dict)
                                    if level_5_dict['id'] == fiche_id:
                                        return get_arbo(arbo)
                            except:
                                print('hello')


def add_data_to_list(data_question, data_list, context, qas):
    if len(data_list) == 0:
        data_question['paragraphs'] = [{'context': context, 'qas': [qas]}]
        data_list = [data_question]
    else:
        for fiche in data_list:
            if fiche['id'] == data_question['id']:
                fiche['paragraphs'][0]['qas'].append(qas)
            else:
                data_question['paragraphs'] = [{'context': context, 'qas': [qas]}]
        data_list.append(data_question)
    return data_list


def get_context_from_url(fiche_id, path_knowledge):
    path_fiche = path_knowledge / f'{fiche_id}.json'
    with open(path_fiche) as file:
        file_content = json.load(file)
    return file_content['text']

def main(path_arbo, path_dataset, path_knowledge):
    with open(path_arbo) as file:
        arborescence = json.load(file)
    df = pd.read_csv(path_dataset)
    df = df[df['url_2'].isnull()] # select only lines with only one url
    dj_json = df.to_json(orient='records')
    dataset = json.loads(dj_json)
    data_list = []
    for question in dataset:
        qas = question['incoming_message']
        url = question['url']
        fiche_id = url.split('/')[-1]
        try:
            context = get_context_from_url(fiche_id, path_knowledge)
            data_question = get_arborescence(arborescence, fiche_id)
            data_list = add_data_to_list(data_question, data_list, context, qas)
        except:
            print(f'the fiche {fiche_id} does not exists !!')
            context = 'none'

    result = {'version': "1.0",
              'data': data_list}

    with open(Path('./data/data_set_annotation.json'), 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    path_arbo = Path('./data/arborescence_spf_particuliers.json')
    path_dataset = Path('./data/407_question-fiche_anonym.csv')
    path_knowledge = Path('./data/v10')
    main(path_arbo, path_dataset, path_knowledge)
