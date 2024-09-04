# Données

Les données utilisées dans l'exercice sont les métadonnées des publications du site de la HAS. Elles sont documentés extensivement sur datagouv.fr : https://www.data.gouv.fr/fr/datasets/metadonnees-des-publications-de-la-has/. Le fichier zip `data/has-publications-single.zip` téléchargé depuis cette url contient les données utilisées dans cet exercice. Il est conseillé d'utiliser le fichier `AllPublications.json` (quitte à tout de suite recaster le json en dataframe) qui parse mieux les champs nestés que la version csv.

Note: Ces données ont été extraites grâce à l'API de la HAS, documentée dans `documentation/documentation_api.md`. **Il n'est pas nécessaire de comprendre le fonctionnement de l'API pour cet exercice.**

## Thématiques

Les thématiques utilisées sur le site internet de la HAS sont décrites dans le document `documentation/categories_thematiques.xlsx`. Elles sont organisées selon une arborescence, que l'on peut reconstruire via l'identifiant du parent. Dans le jeu de données, elles sont décrites dans le champ `categoriesThematiques`.

**NB:** Attention, dans les données, il existe plus de thématiques que celles décrites dans cette documentation. Il pourra être utile de s'intéresser à un sous groupe de thématiques.

## Types de contenus

Les contenus du site de la HAS ont un type, parmi une liste donnée dans le document `documentation/type_contenus.xlsx`. 
Ce document donne également le nom interne (technique) d'un type.

On s'interessera en particulier à catégoriser les documents des types suivants :
- Recommandation de bonne pratique
- Guide maladie chronique
- Guide méthodologique
- Outil d'amélioration des pratiques professionnelles
- Recommandation en santé publique
- Études et Rapports
- Guide usagers
- Recommandation vaccinale
- Avis sur les Médicaments
- Avis sur les dispositifs médicaux et autres produits de santé
- Evaluation des technologies de santé

## Contenus

Pour effectuer la classification, on s'intéressera en priorité au résumé markdown ou html des publications ainsi qu'au titre de la publication (disponibles dans les champs `title` et `resumeSiteWeb`), plutôt qu'aux documents pdf joints (ce qui nécessiterait plus de travail).
