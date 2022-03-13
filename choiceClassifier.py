# This function returns True or False depending on whether the user has entered yes or no.
def choice_classifier(choice, bot_name):
    choice = choice.lower().strip()
    while True:
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            choice = input("[%s]: Please select yes or no.\n[You]: " % bot_name)
