from deployment.roles.haystack.files.custom_component import \
       JoinDocumentsCustom, MergeOverlappingAnswers, StripLeadingSpace, \
       RankAnswersWithWeigth

def join_document_custom(ks_retriever):
    return JoinDocumentsCustom(ks_retriever = ks_retriever)

def merge_overlapping_answers():
    return MergeOverlappingAnswers()

def strip_leading_space():
    return StripLeadingSpace()

def rank_answers_with_weight():
    return RankAnswersWithWeigth()

