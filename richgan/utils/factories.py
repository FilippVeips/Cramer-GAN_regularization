from ..configs.defaults import class_constructor_defaults


def make_factory(classes):
    def factory(classname, **kwargs):
        arguments = {}
        if classname in class_constructor_defaults:
            arguments.update(class_constructor_defaults[classname])
        arguments.update(kwargs)
        return {i_class.__name__: i_class for i_class in classes}[classname](
            **arguments
        )

    return factory
