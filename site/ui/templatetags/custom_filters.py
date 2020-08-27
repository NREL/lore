from django import template
register = template.Library()


@register.filter(name='abs')
def abs_filter(value):
    return abs(value)