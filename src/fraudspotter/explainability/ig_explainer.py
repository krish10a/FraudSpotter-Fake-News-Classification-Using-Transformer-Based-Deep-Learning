from typing import Any, Dict


def explain_with_integrated_gradients(
    text: str,
    model: Any,
    tokenizer: Any,
    target_class: int = 1,
    n_steps: int = 50,
) -> Dict[str, Any]:
    """Generate token-level attributions for DistilBERT using Captum LayerIntegratedGradients."""
    import torch
    from captum.attr import LayerIntegratedGradients

    device = next(model.parameters()).device
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Embedding layer for DistilBERT
    embeddings_layer = model.distilbert.embeddings

    def model_forward(inputs_embeds, attn_mask):
        output = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        return output.logits

    lig = LayerIntegratedGradients(model_forward, embeddings_layer)

    inputs_embeds = embeddings_layer(input_ids)
    ref_input_ids = torch.zeros_like(input_ids)
    ref_inputs_embeds = embeddings_layer(ref_input_ids)

    attributions, delta = lig.attribute(
        inputs=inputs_embeds,
        baselines=ref_inputs_embeds,
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum across embedding dimensions
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_attributions = [
        {"token": str(token), "attribution": float(attr)}
        for token, attr in zip(tokens, attributions.cpu().detach().numpy())
    ]

    return {
        "method": "IntegratedGradients",
        "target_class": target_class,
        "convergence_delta": float(delta.item()) if torch.is_tensor(delta) else float(delta),
        "token_attributions": token_attributions,
    }
