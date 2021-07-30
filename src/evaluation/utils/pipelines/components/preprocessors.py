from haystack.preprocessor.preprocessor import PreProcessor

def preprocessor(split_by, split_length):
    return PreProcessor(
            clean_empty_lines=False,
            clean_whitespace=False,
            clean_header_footer=False,
            split_by=split_by,
            split_length=split_length,
            split_overlap=0,  # this must be set to 0 at the date of writting this: 22 01 2021
            split_respect_sentence_boundary=False,  # the support for this will soon be removed : 29 01 2021
        )

