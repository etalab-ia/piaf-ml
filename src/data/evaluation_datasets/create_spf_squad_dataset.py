"""
WARNING: This script is a one-shot script.
This script creates a single SQuAD format SPF evaluation dataset. This file is the merge of these three elements:
1. The uni-fiche JSON files created from the original SPF xml files (these files contain the dossier,theme metadata)
2. The 105 annotated Question Fiches dataset (data/squad-style-datasets/spf_qr_test.json)
3. The ~530 Question Fiches dataset  (data/questions_spf.json)

The output should resemble this :
{
  "version": "v1.0",
  "data": [{
    'title': 'Immatriculer un véhicule d'occasione',
    'sous_theme': '',
    'categorie': "Carte grise (certificat d'immatriculation)",
    'sous_dossier': 'Immatriculer un véhicule',
    'reference': "Immatriculer un véhicule d'occasion",
    'id': 'F1050',
    'paragraphs': [{
        'context': "Carte grise\xa0: immatriculer un véhicule d'occasion : \n\n\nSi vous achetez une voiture en France ou à l'étranger, vous avez 1 mois pour la faire immatriculer et obtenir ainsi une carte grise (certificat d'immatriculation). En cas de contrôle routier, l'absence de présentation de la carte grise peut entraîner une amende pouvant aller jusqu'à  750\xa0€ . Vous risquez aussi l'immobilisation immédiate du véhicule. La démarche ne s'effectue plus en préfecture mais auprès de l'ANTS, par voie dématérialisée.\nEn ligne\nUn dispositif de copie numérique (scanner, appareil photo numérique, smartphone ou tablette avec fonction photo) est nécessaire. Le format des documents numérisés à transmettre peut être un des suivants\xa0: JPG, PNG, BMP, TIFF, PDF.\nDes  points numériques \n\n (avec ordinateurs, imprimantes et scanners) sont mis à votre disposition dans chaque préfecture et dans la plupart des sous-préfectures.Vous pouvez y accomplir la démarche, aidé par des médiateurs si vous rencontrez des difficultés avec l'utilisation d'internet.\nVous pouvez aussi être accompagné dans votre démarche dans une maison de services au public.\nVous devez faire la démarche sur le site de l'ANTS en vous identifiant via  FranceConnect .\nVous devez disposer d'une copie numérique (photo ou scan) de plusieurs documents  :\nFormulaire  cerfa n°13750 \n\n\n Justificatif de domicile  de moins de 6 mois (ou, en cas de cotitulaires, justificatif de celui dont l'adresse va figurer sur la carte grise)\nCarte grise  du véhicule, barrée avec la mention  Vendu le (jour/mois/année)   ou  Cédé le (jour/mois/année) , et avec la signature de l'ancien propriétaire  (ou de tous les cotitulaires s'il y en avait)\nSi le véhicule a plus de 4 ans, preuve du contrôle technique en cours de validité, sauf si le véhicule en est dispensé . Le contrôle technique doit dater de moins de 6 mois le jour de la demande de carte grise\xa0: si le délai est dépassé, il faudra réaliser un nouveau contrôle à vos frais\nSi vous faites la démarche pour quelqu'un d'autre,  mandat  signé et  pièce d'identité  de la personne pour qui vous effectuez la démarche\nVous devez disposer du  code de cession , remis par l'ancien propriétaire du véhicule.\nVous devez certifier sur l'honneur que le demandeur de la carte grise dispose\nd'une attestation d'assurance du véhicule\net d'un permis de conduire correspondant à la catégorie du véhicule immatriculé.\nVous n'avez pas à joindre une copie numérique (photo ou scan) du permis de conduire. En revanche, celle-ci pourra vous être demandée lors de l'instruction de votre dossier.\nVous devez par ailleurs fournir des informations, notamment\xa0:\nl'identité du titulaire (et éventuellement des co-titulaires) de la carte grise\xa0: nom, prénoms, sexe, date et lieu  de naissance, numéro de téléphone et adresse électronique,\nvos coordonnées bancaires.\nLe règlement du  montant de la carte grise  doit obligatoirement être effectué par carte bancaire.\nLe coût de la carte grise est variable. Il dépend notamment des caractéristiques du véhicule et de la région dans laquelle vous vivez.\nVous pouvez évaluer le coût de votre carte grise en utilisant ce simulateur\xa0:\nÀ la fin de la procédure, vous obtenez\xa0:\nun numéro de dossier,\nun accusé d'enregistrement de votre demande\net un certificat provisoire d'immatriculation (CPI), que vous devez  imprimer. Le CPI vous permet de circuler pendant 1 mois, uniquement en France, en attendant de recevoir votre carte grise.\nVous recevrez votre carte grise sous  pli sécurisé   en général dans les  7  jours ouvrés . Toutefois, le délai peut être plus long\nsi votre demande est incomplète ou doit être analysée par le service instructeur (la fabrication de la carte grise ne pourra être lancée qu'à l'issue de cette étape)\nou en fonction du nombre de demandes en cours de traitement.\nVous pouvez suivre l'état d'avancement de votre dossier sur le site de l'Agence nationale des titres sécurisés (ANTS)\xa0: \nMunissez-vous du certificat provisoire d'immatriculation (CPI).\nSi vous êtes absent lors du passage du facteur, un avis de passage vous sera déposé. Vous avez ensuite 15 jours pour récupérer votre document à La Poste (ou donner procuration à un tiers pour le faire à votre place). Passé ce délai, le titre est retourné à l'expéditeur. Vous devrez contacter l'ANTS pour qu'il vous soit renvoyé.\nPar messagerie\n\n\t\t\t\n      Accès au\n       formulaire en ligne \n\n\t\t\nvous devez conserver l'ancienne carte grise pendant 5 ans, puis la détruire.\nAuprès d'un professionnel habilité\nEn plus du coût de la carte grise, le professionnel vous facturera une somme correspondant à la prestation qu'il réalise à votre place. Cette somme est librement fixée par le professionnel.\nVous devez présenter les documents suivants\xa0: \nCarte grise  du véhicule, barrée avec la mention  Vendu le (jour/mois/année)   ou  Cédé le (jour/mois/année)  ,  avec  la signature de l'ancien propriétaire (ou de tous les co-titulaires s'il y en avait)\nFormulaire  cerfa 13757  de mandat à un professionnel\n\n Justificatif d'identité   (original), un par cotitulaire\n\n Justificatif de domicile  (original ), ou en cas de co-titulaires, le justificatif de celui dont l'adresse va figurer sur la carte grise\n\n Coût de la carte grise , en chèque ou par carte bancaire\n\n Preuve du contrôle technique , si le véhicule a plus de 4 ans et  n'en est pas dispensé . Le contrôle doit avoir moins de 6 mois (2 mois si une  contre-visite  a été prescrite) et doit avoir été réalisé en France. Il doit dater de moins de 6 mois le jour de la demande de carte grise\xa0: si le délai est dépassé, il faudra réaliser un nouveau contrôle à vos frais.\nFormulaire  cerfa n°13750 \n\nFormulaire  cerfa n°15776  de déclaration de cession du véhicule rempli et signé par l'ancien et le nouveau propriétaire (s'il y a plusieurs copropriétaires, chacun doit le signer). Si vous avez perdu le formulaire rempli et signé, vous devrez vous contacter le vendeur pour refaire le formulaire.\nAttestation d'assurance du véhicule à immatriculer\nPermis de conduire correspondant à la catégorie du véhicule à immatriculer\nVous devez par ailleurs fournir des informations, notamment\xa0l'identité du titulaire (et éventuellement des co-titulaires) de la carte grise\xa0: nom, prénoms, sexe, date et lieu  de naissance, numéro de téléphone et adresse électronique.\nUn  certificat provisoire d'immatriculation (CPI)  sera mis à disposition immédiatement.  Il vous permet de circuler pendant un mois (en France uniquement) en attendant de recevoir votre carte grise définitive.\nVous recevrez votre carte grise définitive sous  pli sécurisé  à votre domicile sous  un délai qui peut varier . Vous pouvez  suivre en ligne l'état d'avancement de sa réalisation .\nvous devez conserver l'ancienne carte grise pendant 5 ans, puis la détruire.\nComment faire la démarche\xa0?\nIl n'est désormais plus possible de demander une carte grise auprès de la préfecture ou de la sous-préfecture.\nVous devez faire la démarche sur le site de l'ANTS en vous identifiant via  FranceConnect .\nUn dispositif de copie numérique (scanner, appareil photo numérique, smartphone ou tablette avec fonction photo) est nécessaire. Le format des documents numérisés à transmettre peut être un des suivants\xa0: JPG, PNG, BMP, TIFF, PDF.\nDes  points numériques \n\n (avec ordinateurs, imprimantes et scanners) sont mis à votre disposition dans chaque préfecture et dans la plupart des sous-préfectures.Vous pouvez y accomplir la démarche, aidé par des médiateurs si vous rencontrez des difficultés avec l'utilisation d'internet.\nVous pouvez aussi être accompagné dans votre démarche dans une maison de services au public.\nDans quel délai\xa0?\nVous avez 1 mois pour faire la démarche. Si vous ne faites pas la démarche à temps et que vous êtes contrôlé par les forces de l'ordre,  vous risquez une amende pouvant  aller jusqu'à  750\xa0€  (en général,   amende forfaitaire  de  135\xa0€ ).\nPièces à fournir et obtention du titre\nLa liste des pièces diffère selon que le véhicule était précédemment immatriculé dans un pays de  l'Union européenne  ou dans un autre pays.\nVéhicule immatriculé dans un pays de l'Union européenne\nVous devez vous munir d'une copie numérique (photo ou scan) des documents suivants\xa0: \n Carte grise d'origine, sans mention particulière, ou une pièce officielle de propriété du véhicule\nSi la carte grise a été conservée par les autorités administratives du pays étranger,  document officiel l'indiquant ou certificat international pour automobiles en cours de validité délivré par ces autorités\n\n Justificatif de domicile  de moins de 6 mois (ou, en cas de cotitulaires, justificatif de celui dont l'adresse va figurer sur la carte grise)\nFormulaire  cerfa n°13750 \n\n\n Preuve du contrôle technique , si le véhicule a plus de 4 ans et  n'en est pas dispensé . Le contrôle doit avoir moins de 6 mois (quand une  contre-visite  a été prescrite, le délai accordé pour l'effectuer ne doit pas être dépassé) et doit avoir été réalisé en France ou dans l'Union européenne si le véhicule y était immatriculé. Le contrôle technique doit dater de moins de 6 mois le jour de la demande de carte grise\xa0: si le délai est dépassé, il faudra réaliser un nouveau contrôle à vos frais.\nSauf si la demande d'immatriculation est revêtue d'une mention de dispense attribuée par les services fiscaux,  quitus fiscal  délivré par la recette principale des impôts attestant que la TVA a bien été payée en France. Le quitus n'est pas à fournir pour une remorque ou semi-remorque.\nSi vous faites la démarche pour quelqu'un d'autre,  mandat  signé et  pièce d'identité  de la personne pour qui vous effectuez la démarche\nSi l'ancienne carte grise ne peut pas être fournie ou ne correspond pas au véhicule importé ou ne permet pas de l'identifier ou ne comporte pas toutes les données obligatoires, justificatif complémentaire correspondant à votre situation :\nCertificat de conformité européen délivré par le constructeur, édité si nécessaire dans une autre langue que le français. Il peut être délivré sous forme de document numérique.\nAttestation d'identification à un type communautaire\nProcès-verbal de réception à titre isolé (RTI) établi par une  Dreal .\nSi vous habitez en  Île-de-France , vous devez demander le procès-verbal de réception à titre isolé -RTI -à la plate-forme régionale de réception de véhicules de la  DRIEE Ile de France .\nVous devez certifier sur l'honneur que le demandeur de la carte grise dispose\nd'une attestation d'assurance du véhicule\net d'un permis de conduire correspondant à la catégorie du véhicule immatriculé.\nVous n'avez pas à joindre une copie numérique (photo ou scan) du permis de conduire. En revanche, celle-ci pourra vous être demandée lors de l'instruction de votre dossier.\nVous devez par ailleurs fournir des informations, notamment\xa0:\nl'identité du titulaire (et éventuellement des cotitulaires) de la carte grise\xa0: nom, prénoms, sexe, date et lieu  de naissance, numéro de téléphone et adresse électronique,\nvos coordonnées bancaires.\nLe règlement du  montant de la carte grise  doit obligatoirement être effectué par carte bancaire.\nLe coût de la carte grise est variable. Il dépend notamment des caractéristiques du véhicule et de la région dans laquelle vous vivez.\nVous pouvez évaluer le coût de votre carte grise en utilisant ce simulateur\xa0:\nÀ la fin de la procédure, vous obtenez\xa0:\nun numéro de dossier,\nun accusé d'enregistrement de votre demande\net un certificat provisoire d'immatriculation (CPI), que vous devez  imprimer. Le CPI vous permet de circuler pendant 1 mois, uniquement en France, en attendant de recevoir votre carte grise.\nVous recevrez votre carte grise sous  pli sécurisé   en général dans les  7  jours ouvrés . Toutefois, le délai peut être plus long\nsi votre demande est incomplète ou doit être analysée par le service instructeur (la fabrication de la carte grise ne pourra être lancée qu'à l'issue de cette étape)\nou en fonction du nombre de demandes en cours de traitement.\nVous pouvez suivre l'état d'avancement de votre dossier sur le site de l'Agence nationale des titres sécurisés (ANTS)\xa0: \nMunissez-vous du certificat provisoire d'immatriculation (CPI).\nSi vous êtes absent lors du passage du facteur, un avis de passage vous sera déposé. Vous avez ensuite 15 jours pour récupérer votre document à La Poste (ou donner procuration à un tiers pour le faire à votre place). Passé ce délai, le titre est retourné à l'expéditeur. Vous devrez contacter l'ANTS pour qu'il vous soit renvoyé.\nDans un autre pays\nVous devez vous munir d'une copie numérique (photo ou scan) des documents suivants\xa0:\n Carte grise d'origine, sans mention particulière, ou pièce officielle de propriété du véhicule\nSi la carte grise a été conservée par les autorités administratives du pays étranger,  document officiel l'indiquant, ou certificat international pour automobiles en cours de validité délivré par ces autorités\n\n Justificatif de domicile  de moins de 6 mois (ou, en cas de cotitulaires, justificatif de celui dont l'adresse va figurer sur la carte grise)\nFormulaire  cerfa n°13750 \n\n\n Preuve du contrôle technique , si le véhicule a plus de 4 ans et  n'en est pas dispensé . Le contrôle doit avoir moins de 6 mois (quand une  contre-visite  a été prescrite, le délai accordé pour l'effectuer ne doit pas être dépassé) et doit avoir été réalisé en France ou dans un pays de l'Union européenne. Le contrôle technique doit dater de moins de 6 mois le jour de la demande de carte grise\xa0: si le délai est dépassé, il faudra réaliser un nouveau contrôle à vos frais.\nCertificat de dédouanement 846 A, sauf si la demande d'immatriculation est revêtue d'une mention de dispense attribuée par les services des douanes\nSi vous faites la démarche pour quelqu'un d'autre,  mandat  signé et  pièce d'identité  de la personne pour qui vous effectuez la démarche\nJustificatif technique de conformité correspondant à la situation du véhicule :\ncertificat de conformité européen, délivré par le constructeur\nou attestation d'identification du véhicule au type communautaire, délivrée par le constructeur ou son représentant en France ou une  Dreal \n\nou procès-verbal de réception à titre isolé (RTI) délivré par une  Dreal \n\nSi vous habitez en  Île-de-France , vous devez demander le procès-verbal de RTI à la plate-forme régionale de réception de véhicules de la  DRIEE Ile de France .\nVous devez certifier sur l'honneur que le demandeur de la carte grise dispose\nd'une attestation d'assurance du véhicule\net d'un permis de conduire correspondant à la catégorie du véhicule immatriculé.\nVous n'avez pas à joindre une copie numérique (photo ou scan) du permis de conduire. En revanche, celle-ci pourra vous être demandée lors de l'instruction de votre dossier.\nVous devez par ailleurs fournir des informations, notamment:\nl'identité du titulaire (et éventuellement des co-titulaires) de la carte grise\xa0: nom, prénoms, sexe, date et lieu  de naissance, numéro de téléphone et adresse électronique,\nvos coordonnées bancaires.\nLe règlement du  montant de la carte grise  doit obligatoirement être effectué par carte bancaire.\nLe coût de la carte grise est variable. Il dépend notamment des caractéristiques du véhicule et de la région dans laquelle vous vivez.\nVous pouvez évaluer le coût de votre carte grise en utilisant ce simulateur\xa0:\nÀ la fin de la procédure, vous obtenez\xa0:\nun numéro de dossier,\nun accusé d'enregistrement de votre demande\net un certificat provisoire d'immatriculation (CPI), que vous devez  imprimer. Le CPI vous permet de circuler pendant 1 mois, uniquement en France, en attendant de recevoir votre carte grise.\nVous recevrez votre carte grise sous  pli sécurisé   en général dans les  7  jours ouvrés . Toutefois, le délai peut être plus long\nsi votre demande est incomplète ou doit être analysée par le service instructeur (la fabrication de la carte grise ne pourra être lancée qu'à l'issue de cette étape)\nou en fonction du nombre de demandes en cours de traitement.\nVous pouvez suivre l'état d'avancement de votre dossier sur le site de l'Agence nationale des titres sécurisés (ANTS)\xa0: \nMunissez-vous du certificat provisoire d'immatriculation (CPI).\nSi vous êtes absent lors du passage du facteur, un avis de passage vous sera déposé. Vous avez ensuite 15 jours pour récupérer votre document à La Poste (ou donner procuration à un tiers pour le faire à votre place). Passé ce délai, le titre est retourné à l'expéditeur. Vous devrez contacter l'ANTS pour qu'il vous soit renvoyé.\nCoût\nLe coût de la carte grise est variable. Il dépend notamment des caractéristiques du véhicule et de la région dans laquelle vous vivez.\nVous pouvez évaluer le coût de votre carte grise en utilisant ce simulateur\xa0:\nServeur vocal interactif national apportant des réponses automatisées concernant la carte grise, le permis de conduire, la carte nationale d'identité et le passeport. \nSi l'usager n'obtient pas la réponse à sa question relative à la carte grise ou au permis de conduire, il sera mis en relation avec un conseiller de l'Agence nationale des titres sécurisés (ANTS).\nPar téléphone\nDepuis la métropole\xa0:\n\n 34 00  (coût d'un appel local)\nDepuis l'outre-mer\xa0:\n\n 09 70 83 07 07 \n\nDepuis l'étranger\xa0:\n\n +33 9 70 83 07 07 \n\nAllemagne, Autriche, Belgique, Bulgarie, Chypre, Croatie, Danemark, Espagne, Estonie, Finlande, France, Grèce, Hongrie, Irlande, Italie, Lituanie, Lettonie, Luxembourg, Malte, Pays-Bas, Pologne, Portugal, République tchèque, Roumanie,  Slovaquie, Slovénie, Suède. Attention\xa0: le Royaume-uni a quitté  l'Union européenne, mais le droit européen concernant les citoyens  s'applique jusqu'au 31 décembre 2020.\nJour effectivement travaillé dans une entreprise ou une administration.\n\t\tOn en compte 5 par semaine.\nSecond examen nécessaire pour vérifier si les points défectueux détectés (défaillance majeure ou critique) lors de la visite initiale du véhicule ont été réparés\nCourrier suivi remis contre signature\nCode informatique obtenu par le vendeur particulier à l'issue de la démarche de la téléprocédure de la vente d'un véhicule d'occasion. Il doit être communiqué au nouveau propriétaire du véhicule afin qu'il réalise sa demande de carte grise.\nConnexion avec l'identifiant et le mot de passe de votre compte Impots.gouv.fr ou Ameli.fr ou Iidentitenumerique.laposte.fr ou Mobileconnectetmoi.fr ou Msa.fr ou Alicem\nDépartements 75, 77, 78, 91, 92, 93, 94 et 95\nDirection régionale de l'environnement, de l'aménagement et du logement\nDirection régionale et interdépartementale de l'environnement et de l'énergie",
        'qas': [
            {'question':"J'ai acheter un véhicule mais j'ai égarer le certificat de cession je dois faire la carte grise a mon nom et je n'arrive pas a joindre le vendeur comment dois je faire?"
             'answers': [ ... ]
             "is_impossible": False

            }]
        }]
    }]

Usage:
    create_spf_squad_dataset.py <spf_fiches_folder> <annotated_questions_spf>

Arguments:
    <spf_fiches_folder>             A path where to find the list of [fiche_id, question, answer] to transform
    <annotated_questions_spf>       Path to the SQuAD JSON file with the annotated spf questions
"""

import glob
import json

from argopt import argopt
from pathlib import Path


def format_qas_as_squad(fiche_id, context, question, answer_start, answer_text):
    """
    Once all parameters are found, formats in SQuAD-like output.
    """
    res = {
        'title': fiche_id,
        'paragraphs': [
            {
                'context': context,
                'qas': [
                    {
                        'question': question,
                        'answers': [
                            {
                                'answer_start': answer_start,
                                'text': answer_text
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return res


def format_context_as_squad(fiche_id, context):
    """
    For fiches which have no question, add them without qas.
    """
    res = {
        'title': fiche_id,
        'paragraphs': [
            {
                'context': context,
            }
        ]
    }
    return res


def create_squad_dataset(spf_fiches_folder: Path,
                         annotated_questions_spf_path: Path):
    # 1. Read all the fiches JSON files
    spf_jsons_paths = [Path(f) for f in glob.glob(spf_fiches_folder.as_posix() + "/*.json")]
    dict_spf_jsons = {}
    for path in spf_jsons_paths:
        with open(path) as filo:
            dict_spf_jsons[path.stem] = json.load(filo)

    # 2. Read all the 105 answered questions + 525 non-answered questions from the annotated dataset
    # Get titles of aready answered questions and of all questions
    with open(annotated_questions_spf_path) as filo:
        annotated_questions_spf = json.load(filo)["data"]
    non_answered_question_fiches_ids = []
    answered_questions_fiches_ids = []
    dict_question_spf = {}
    for fiche in annotated_questions_spf:
        dict_question_spf[fiche["title"]] = fiche
        for paragraph in fiche["paragraphs"]:
            if "qas" in paragraph:
                answered_questions_fiches_ids.append(fiche["title"])
                continue
            else:
                non_answered_question_fiches_ids.append(fiche['title'])
    all_questions_fiches = answered_questions_fiches_ids + non_answered_question_fiches_ids

    non_question_fiches = [f for f in list(dict_spf_jsons.keys()) if f not in all_questions_fiches]
    # 3. Begin with the creation of a SQuAD forma dataset from SPF jsons
    spf_data = []
    for fiche_id in non_question_fiches + all_questions_fiches:
        fiche_content = dict_spf_jsons[fiche_id]
        # if fiche_id in all_questions_fiches:
        #     print("stop!")  # we will add question fiches at the end
        squad_dict = {"link": fiche_content["link"]}

        if len(fiche_content["text"]) < 30:
            print(f"Fiche {fiche_id} has text too small: {fiche_content['text']}")
            continue
        if "arborescence" in fiche_content and fiche_content["arborescence"]:
            squad_dict["title"] = fiche_content["arborescence"].pop("fiche")
            squad_dict.update(fiche_content["arborescence"])
        else:
            print(f"Fiche {fiche_id} has no arborescence")
            squad_dict["title"] = fiche_content["text"].split("\n")[0][:-3]
            squad_dict.update({
                'sous_theme': '',
                'categorie': '',
                'sous_dossier': '',
                'reference': squad_dict["title"],
                'id': fiche_id
            })

        # if fiche_id not in non_answered_question_fiches_ids:
        #     question = ""
        # if fiche_id in non_answered_question_fiches_ids:  # this fiche is a question fiche without answer
        #     question = squad_dict["title"]
        # else:
        #     question

        squad_dict["paragraphs"] = [
            {
                "context": fiche_content["text"],
                "qas": [
                    {
                        "question": "" if fiche_id not in non_answered_question_fiches_ids else squad_dict["title"],
                        "answers": [{"answer_start": -1, "text": ""}] if fiche_id not in answered_questions_fiches_ids else
                        dict_question_spf[fiche_id]["paragraphs"][0]["qas"][0]["answers"],
                        "is_impossible": False
                    }
                ]
            }
        ]
        spf_data.append(squad_dict)

    # 4. Save the new dataset
    new_dataset = {"version": 1.0,
                   "data": spf_data}
    with open(annotated_questions_spf_path.parent / Path("full_spf_squad.json"), "w") as filo:
        json.dump(new_dataset, filo, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argopt(__doc__).parse_args()
    spf_fiches_folder = Path(parser.spf_fiches_folder)
    annotated_questions_spf_path = Path(parser.annotated_questions_spf)
    create_squad_dataset(spf_fiches_folder=spf_fiches_folder,
                         annotated_questions_spf_path=annotated_questions_spf_path)
