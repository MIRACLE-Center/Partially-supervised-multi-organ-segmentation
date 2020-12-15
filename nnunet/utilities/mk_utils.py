def get_length(generator):
    if hasattr(generator,"__len__"):
        return len(generator)
    else:
        return sum(1 for _ in generator)


def get_tag_index(tag):

    tasks = ['liver','spleen','pancreas','rightkidney','leftkidney']
    return tasks.index(tag)+1