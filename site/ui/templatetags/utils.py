from django import template
register = template.Library()

@register.filter(name='avgcalc')
def avgcalc(num_array):

    if type(num_array) != list:
        return num_array
    
    return sum(num_array)/len(num_array)