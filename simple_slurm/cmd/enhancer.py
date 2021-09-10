def Conditional(param: str, cond: dict):
    """
    apply conditional logic on the class

    Args:
        param (str): param name that is conditioned on
        cond (dict): key = multiple value of `param`, one of the keys must exist in the input of the decorated class
                     value = dict, param_name -> param_value

    Returns:
        func that takes input of class and built the decoarted class
    """
    # cls = class to apply Condition on
    def wrapper(cls):
        # cls takes input as kwargs
        def cls_init(*args, **kwargs):
            assert param in kwargs
            assert kwargs[param] in cond
            
            kwargs.update(cond[kwargs[param]])
            
            return cls(*args, **kwargs)

        return cls_init

    return wrapper

if __name__ == "__main__":
    from .SalKGCommand import FineCommand
    
    @Conditional(
        param="dataset",
        cond=dict(
            csqa=dict(
                text="roberta_large",
                graph="mhgrn"
            ),
            obqa=dict(
                text="bert_base_uncased",
                graph="pathgen"
            )
        )
    )
    fc = FineCommand(
        dataset="csqa",
        tlr=[1e-5, 1e-3], 
        glr=[3e-3, 2e-3], 
        wd=0.02, 
        pos_w=10, 
        seed=[0, 1, 2], 
        save_checkpoint=True,
        save_saliency=True
    )
    print(fc.mkString)